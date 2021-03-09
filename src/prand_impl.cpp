#include <memory>
#include <sstream>
#include <glog/logging.h>
#include <cuda_runtime.h>

extern "C" {
#include <libavutil/pixdesc.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>
}

#include "NvDecoder/ColorSpace.h"
#include "prand_impl.hpp"

struct AVINIT {
	AVINIT() {
		::av_register_all();
		CHECK_GE(::avformat_network_init(), 0);
	}
	~AVINIT() {
		CHECK_GE(::avformat_network_deinit(), 0);
	}
} g_AVInit;

inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID codecID) {
	switch (codecID) {
	case AV_CODEC_ID_MPEG1VIDEO:	return cudaVideoCodec_MPEG1;
	case AV_CODEC_ID_MPEG2VIDEO:	return cudaVideoCodec_MPEG2;
	case AV_CODEC_ID_MPEG4:			return cudaVideoCodec_MPEG4;
	case AV_CODEC_ID_VC1:			return cudaVideoCodec_VC1;
	case AV_CODEC_ID_H264:			return cudaVideoCodec_H264;
	case AV_CODEC_ID_HEVC:			return cudaVideoCodec_HEVC;
	case AV_CODEC_ID_VP8:			return cudaVideoCodec_VP8;
	case AV_CODEC_ID_VP9:			return cudaVideoCodec_VP9;
	case AV_CODEC_ID_MJPEG:			return cudaVideoCodec_JPEG;
	default:						return cudaVideoCodec_NumCodecs;
	}
}

void DestroyCudaContext(CUcontext *pCuCtx) {
	CUDA_DRVAPI_CALL(cuCtxDestroy(*pCuCtx));
	delete pCuCtx;
}

std::shared_ptr<CUcontext> MakeCudaContext(int nGpuID) {
	CUdevice cuDev;
	CUDA_DRVAPI_CALL(cuDeviceGet(&cuDev, nGpuID));
	CUcontext *pCuCtx = new CUcontext;
	CUDA_DRVAPI_CALL(cuCtxCreate(pCuCtx, 0, cuDev));
	return std::shared_ptr<CUcontext>(pCuCtx, &DestroyCudaContext);
}

void DestroyAVContext(AVFormatContext *pAVCtx) {
	if (pAVCtx) {
		::avformat_close_input(&pAVCtx);
	}
}

void DestroyAVBsfc(AVBSFContext *pAVBsfc) {
	if (pAVBsfc) {
		::av_bsf_free(&pAVBsfc);
	}
}

PrandImpl::PrandImpl(int nGpuID)
		: m_nGpuID(nGpuID)
		, m_nFrameCnt(0)
		, m_Status(STATUS::STANDBY) {
	m_FilterPacket.data = nullptr;
	m_FilterPacket.size = 0;

	// CUDA Device Initialization
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));
	CUDA_CHECK(::cudaStreamCreate(&m_CudaStream));

	// JPEG Encoder Initialization
	NVJPEG_CHECK(::nvjpegCreateSimple(&m_JpegHandle));
	NVJPEG_CHECK(::nvjpegEncoderStateCreate(m_JpegHandle,
			&m_JpegState, m_CudaStream));
	NVJPEG_CHECK(::nvjpegEncoderParamsCreate(m_JpegHandle,
			&m_JpegParams, m_CudaStream));

	NVJPEG_CHECK(::nvjpegEncoderParamsSetSamplingFactors(m_JpegParams,
			NVJPEG_CSS_444, m_CudaStream));
#if CUDA_VERSION_MINOR > 1
	NVJPEG_CHECK(::nvjpegEncoderParamsSetEncoding(m_JpegParams,
			NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, m_CudaStream));
#endif
	NVJPEG_CHECK(::nvjpegEncoderParamsSetQuality(m_JpegParams,
			80, m_CudaStream));
	NVJPEG_CHECK(::nvjpegEncoderParamsSetOptimizedHuffman(m_JpegParams,
			1, m_CudaStream));

	m_pCuCtx = ::MakeCudaContext(m_nGpuID);
	CHECK_NOTNULL(m_pCuCtx.get());
}

PrandImpl::~PrandImpl() {
	Stop();
	if (m_FilterPacket.data != nullptr) {
		av_packet_unref(&m_FilterPacket);
	}
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));
	NVJPEG_CHECK(::nvjpegEncoderParamsDestroy(m_JpegParams));
	NVJPEG_CHECK(::nvjpegEncoderStateDestroy(m_JpegState));
	NVJPEG_CHECK(::nvjpegDestroy(m_JpegHandle));
	CUDA_CHECK(::cudaStreamDestroy(m_CudaStream));
}

std::pair<bool, cv::Size> PrandImpl::Start(const std::string &strURL,
		READ_MODE readMode) {
	CHECK(m_Status == STATUS::STANDBY);

	// Creating AV Context
	// -------------------
	AVDictionary *pDict = nullptr;
	CHECK_GE(::av_dict_set(&pDict, "rtsp_transport", "tcp", 0), 0);
	AVFormatContext *pAVCtxRaw = nullptr;
	if (::avformat_open_input(&pAVCtxRaw, strURL.c_str(),
			nullptr, &pDict) != 0) {
#ifdef NDEBUG
		LOG(WARNING) << "Can't open stream: \"" << strURL << "\"";
#endif
		return std::make_pair(false, cv::Size());
	}
	m_pAVCtx.reset(pAVCtxRaw, &::DestroyAVContext);
	::av_dict_free(&pDict);

	// Find the Best Stream in AV Context
	// ----------------------------------
	CHECK_GE(::avformat_find_stream_info(m_pAVCtx.get(), nullptr), 0);
	AVCodec *pAVDecoder = nullptr;
	m_nStreamId = ::av_find_best_stream(m_pAVCtx.get(), AVMEDIA_TYPE_VIDEO,
			-1, -1, &pAVDecoder, 0);
	CHECK_GE(m_nStreamId, 0);
	CHECK_NOTNULL(pAVDecoder);
	AVStream *pStream = m_pAVCtx->streams[m_nStreamId];

	// Creating NvDecoder with Best Stream
	// -----------------------------------
	m_CurCodecId = ::FFmpeg2NvCodecId(pAVDecoder->id);
	m_pDecoder.reset(new NvDecoder(*m_pCuCtx.get(), true, m_CurCodecId, false));

	// Initializing AV Bit Stream Filter for H.264/H.265
	// -------------------------------------------------
	const AVBitStreamFilter *pBsf = nullptr;
	if (m_CurCodecId == cudaVideoCodec_H264) {
		pBsf = av_bsf_get_by_name("h264_mp4toannexb");
	} else if (m_CurCodecId == cudaVideoCodec_HEVC) {
		pBsf = av_bsf_get_by_name("hevc_mp4toannexb");
	}
	if (pBsf != nullptr) {
		AVBSFContext *pBsfc = nullptr;
		CHECK_EQ(av_bsf_alloc(pBsf, &pBsfc), 0);
		m_pAVBsfc.reset(pBsfc, &::DestroyAVBsfc);
		avcodec_parameters_copy(pBsfc->par_in, pStream->codecpar);
		CHECK_EQ(av_bsf_init(pBsfc), 0);
	}
	if (m_FilterPacket.data != nullptr) {
		av_packet_unref(&m_FilterPacket);
	}
	av_init_packet(&m_FilterPacket);
	m_FilterPacket.data = nullptr;
	m_FilterPacket.size = 0;

	// Initializing Internal Variables
	// -------------------------------
	m_nFrameCnt = 0;
	m_Status = STATUS::WORKING;
	m_dTimeBase = av_q2d(pStream->time_base);
	if (readMode == READ_MODE::AUTO) { // AUTO: block for file, async for rtsp
		m_bBlocking = (strURL.find("rtsp://", 0) != 0);
	} else {
		m_bBlocking = (readMode == READ_MODE::BLOCK);
		CHECK(readMode == READ_MODE::ASYNC || readMode == READ_MODE::BLOCK);
	}
#ifdef NDEBUG
	cv::Size frameSize(pStream->codecpar->width, pStream->codecpar->height);
	LOG(INFO) << "Decoder: " << m_CurCodecId << ", resolution: " << frameSize
			  << ", Blocking: " << m_bBlocking;
#endif

	// Streaming Started
	// -----------------
	m_Worker = std::thread(&PrandImpl::__WorkerProc, this);
	return std::make_pair(true, frameSize);
}

// Stop streaming and clear the FAILED status
void PrandImpl::Stop() {
	m_Status = STATUS::STANDBY;
	if (m_Worker.joinable()) {
		m_Worker.join();
	}
	m_bBlocking = false;
}

// Return Value: STATUS::WORKING if it working normally regardless of whether
//   it is in blocking mode or not. STATUS::STANDBY if it is not started yet or
//   streaming to end of the media. STATUS::FAILED if any error occured.
PrandImpl::STATUS PrandImpl::GetCurrentStatus() const {
	if (m_Worker.joinable()) {
		return STATUS::WORKING;
	}
	return m_Status;
}

// Return Value: greater or equel to zero if it working normally,
//   otherwise it is failure. Please note that if the returned value
//   equal to zero or is the save as the previous, the `frameImg` remains
//   unchanged and the caller should retry to get the next frame.
int64_t PrandImpl::GetFrame(cv::cuda::GpuMat &frameImg, std::string *pJpegData) {
	CUDA_CHECK(cudaSetDevice(m_nGpuID));
	if (m_bBlocking) {
		if (m_Worker.joinable()) {
			m_Worker.join();
		}
	}

	int64_t nFrameCnt = -1;
	if (m_Status == STATUS::WORKING) {
		cv::cuda::GpuMat outImg;
		m_Mutex.lock();
		if (!m_WorkingBuf.empty()) {
			m_WorkingBuf.copyTo(outImg);
		}
		nFrameCnt = m_nFrameCnt;
		m_Mutex.unlock();
		if (m_bBlocking) {
			m_Worker = std::thread(&PrandImpl::__WorkerProc, this);
		}
		if (!outImg.empty()) {
			frameImg.swap(outImg);
			if (pJpegData != nullptr) {
				__EncodeJPEG(frameImg, pJpegData);
			}
		}
	}
	return nFrameCnt;
}

void PrandImpl::__WorkerProc() {
	const int64_t nUserTimeScale = 1000;
	AVPacket packet;
	for (; m_Status == STATUS::WORKING; ) {
		av_init_packet(&packet);
		int64_t nRet = -1;
		for (; ; ::av_packet_unref(&packet)) {
			nRet = ::av_read_frame(m_pAVCtx.get(), &packet);
			if (nRet < 0 || packet.stream_index == m_nStreamId) {
				break;
			}
		}
		int64_t nDecodedFrames = 0;
		if (nRet >= 0) {
			uint8_t *pData = packet.data;
			int nSize = packet.size;
			int64_t pts = packet.pts * nUserTimeScale * m_dTimeBase;
			if (m_CurCodecId == cudaVideoCodec_H264 || m_CurCodecId == cudaVideoCodec_HEVC) {
				CHECK_EQ(av_bsf_send_packet(m_pAVBsfc.get(), &packet), 0);
				if (m_FilterPacket.data) {
					av_packet_unref(&m_FilterPacket);
				}
				CHECK_EQ(av_bsf_receive_packet(m_pAVBsfc.get(), &m_FilterPacket), 0);
				pData = m_FilterPacket.data;
				nSize = m_FilterPacket.size;
				pts = m_FilterPacket.pts * nUserTimeScale * m_dTimeBase;
			}
			nDecodedFrames = m_pDecoder->Decode(pData, nSize, 0, pts);
			if (nDecodedFrames) {
				std::lock_guard<std::mutex> locker(m_Mutex);
				__DecodeFrame(packet, m_WorkingBuf);
				m_nFrameCnt += nDecodedFrames;
			}
		} else if (nRet == AVERROR_EOF) { // EOF encountered,
			m_Status = STATUS::STANDBY; // streaming stopped  spontaneously
		} else {
#ifdef NDEBUG
			std::string strMsg(1024, '\0');
			auto nErr = av_strerror(nRet, (char *)strMsg.data(), strMsg.size());
			if (nErr < 0) {
				strMsg = "UNKNOWN";
			}
			LOG(WARNING) << "One frame lost, code=" << nRet
						 << ", message=\"" << strMsg << "\"";
#endif
			m_Status = STATUS::FAILED;
		}
		av_packet_unref(&packet);
		if (m_bBlocking && nDecodedFrames > 0) {
			break;
		}
	}
}

void PrandImpl::SetJpegQuality(int nQuality) {
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));
	CHECK_GT(nQuality, 0);
	CHECK_LE(nQuality, 100);
	NVJPEG_CHECK(::nvjpegEncoderParamsSetQuality(m_JpegParams,
			nQuality, m_CudaStream));
}

void PrandImpl::__DecodeFrame(const AVPacket &packet, cv::cuda::GpuMat &gpuImg) {
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));

	auto frameFormat = m_pDecoder->GetOutputFormat();
	cv::Size imgSize(m_pDecoder->GetWidth(), m_pDecoder->GetHeight());
	size_t nPitch = imgSize.width * 4;
	size_t nImgBytes = imgSize.height * nPitch;
	uint8_t iMatrix = m_pDecoder->GetVideoFormatInfo()
			.video_signal_description.matrix_coefficients;
	uint8_t *pSrc = m_pDecoder->GetFrame();
	CHECK_NOTNULL(pSrc);

	if (m_BGRATmp.size() != imgSize || m_BGRATmp.channels() != 4
			|| m_BGRATmp.type() != CV_8UC4) {
		m_BGRATmp = cv::cuda::GpuMat(imgSize, CV_8UC4);
	}
	uint8_t *pRGBATmp = m_BGRATmp.ptr<uint8_t>();
	if (m_pDecoder->GetBitDepth() == 8) {
		if (frameFormat == cudaVideoSurfaceFormat_YUV444) {
			::YUV444ToColor32<BGRA32>(pSrc, imgSize.width, pRGBATmp,
					nPitch, imgSize.width, imgSize.height, iMatrix);
		} else {
			::Nv12ToColor32<BGRA32>(pSrc, imgSize.width, pRGBATmp,
					nPitch, imgSize.width, imgSize.height, iMatrix);
		}
	} else {
		if (frameFormat == cudaVideoSurfaceFormat_YUV444) {
			::YUV444P16ToColor32<BGRA32>(pSrc, 2 * imgSize.width, pRGBATmp,
					nPitch, imgSize.width, imgSize.height, iMatrix);
		} else {
			::P016ToColor32<BGRA32>(pSrc, 2 * imgSize.width, pRGBATmp,
					nPitch, imgSize.width, imgSize.height, iMatrix);
		}
	}
	if (gpuImg.size() != imgSize || gpuImg.channels() != 3
			|| gpuImg.type() != CV_8UC3) {
		gpuImg = cv::cuda::GpuMat(imgSize, CV_8UC3);
	}
	::BGRA32ToBgr24(m_BGRATmp.data, gpuImg.data, imgSize.width, imgSize.height,
			gpuImg.step);
}

void PrandImpl::__EncodeJPEG(cv::cuda::GpuMat &frameImg, std::string *pJpegData) {
	nvjpegImage_t nvImg = { 0 };
	nvImg.channel[0] = frameImg.data;
	nvImg.pitch[0] = frameImg.step;
	NVJPEG_CHECK(::nvjpegEncodeImage(m_JpegHandle, m_JpegState,
			m_JpegParams, &nvImg, NVJPEG_INPUT_BGRI,
			frameImg.cols, frameImg.rows, m_CudaStream));
	size_t nSize = 0;
	NVJPEG_CHECK(::nvjpegEncodeRetrieveBitstream(m_JpegHandle,
			m_JpegState, NULL, &nSize, m_CudaStream));
	CHECK_GT(nSize, 0);
	pJpegData->resize(nSize);
	CUDA_CHECK(::cudaStreamSynchronize(m_CudaStream));
	NVJPEG_CHECK(::nvjpegEncodeRetrieveBitstream(m_JpegHandle, 
			m_JpegState, (uint8_t*)(pJpegData->data()), &nSize, 0));
	CUDA_CHECK(::cudaStreamSynchronize(m_CudaStream));
}