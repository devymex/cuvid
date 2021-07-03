#include <memory>
#include <sstream>
#include "logging.hpp"
#include <cuda_runtime.h>

extern "C" {
#include <libavutil/pixdesc.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>
}

#include "NvDecoder/ColorSpace.h"
#include "cuvid_impl.hpp"

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

CuvidImpl::CuvidImpl(int nGpuID)
		: m_nGpuID(nGpuID)
		, m_nFrameCnt(0)
		, m_nCursor(0)
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

CuvidImpl::~CuvidImpl() {
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

bool CuvidImpl::Start(const std::string &strURL,
		READ_MODE readMode) {
	CHECK(m_Status == STATUS::STANDBY);

	// Creating AV Context
	// -------------------
	AVDictionary *pDict = nullptr;
	CHECK_GE(::av_dict_set(&pDict, "rtsp_transport", "tcp", 0), 0);
	AVFormatContext *pAVCtxRaw = nullptr;
	if (::avformat_open_input(&pAVCtxRaw, strURL.c_str(),
			nullptr, &pDict) != 0) {
#ifdef VERBOSE_LOG
		LOG(WARNING) << "Can't open stream: \"" << strURL << "\"";
#endif
		return false;
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
	} else {
		CHECK(m_CurCodecId == cudaVideoCodec_H264 || m_CurCodecId == cudaVideoCodec_HEVC);
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
	m_nCursor = 0;
	m_bEOF = false;
	m_Status = STATUS::WORKING;
	m_dTimeBase = av_q2d(pStream->time_base);
	if (readMode == READ_MODE::AUTO) { // AUTO: block for file, async for rtsp
		m_bBlocking = (strURL.find("rtsp://", 0) != 0);
	} else {
		m_bBlocking = (readMode == READ_MODE::BLOCK);
		CHECK(readMode == READ_MODE::ASYNC || readMode == READ_MODE::BLOCK);
	}
	cv::Size frameSize(pStream->codecpar->width, pStream->codecpar->height);
#ifdef VERBOSE_LOG
	LOG(INFO) << "Decoder: " << m_CurCodecId << ", resolution: " << frameSize
			  << ", Blocking: " << m_bBlocking;
#endif

	// Streaming Started
	// -----------------
	m_Worker = std::thread(&CuvidImpl::__WorkerProc, this);
	return true;
}

// Stop streaming and clear the FAILED status
void CuvidImpl::Stop() {
	m_Status = STATUS::STANDBY;
	if (m_Worker.joinable()) {
		m_Worker.join();
	}
	m_bBlocking = false;
}

double CuvidImpl::get(cv::VideoCaptureProperties prop) const {
	CHECK_NOTNULL(m_pAVCtx);
	AVStream *pStream = m_pAVCtx->streams[m_nStreamId];
	if (prop == cv::CAP_PROP_FPS) {
		auto fr = pStream->avg_frame_rate;
		return (double)fr.num / (double)fr.den;
	} else if (prop == cv::CAP_PROP_FRAME_COUNT) {
		return pStream->nb_frames;
	} else if (prop == cv::CAP_PROP_FRAME_WIDTH) {
		return pStream->codecpar->width;
	} else if (prop == cv::CAP_PROP_FRAME_HEIGHT) {
		return pStream->codecpar->height;
	}
	LOG(FATAL) << "Unsupported prop!";
	return 0.f;
}

// Return Value: STATUS::WORKING if it working normally regardless of whether
//   it is in blocking mode or not. STATUS::STANDBY if it is not started yet or
//   streaming to end of the media. STATUS::FAILED if any error occured.
CuvidImpl::STATUS CuvidImpl::GetCurrentStatus() const {
	if (m_Worker.joinable()) {
		return STATUS::WORKING;
	}
	return m_Status;
}

// Return Value: greater or equel to zero if it working normally,
//   otherwise it is failure. Please note that if the returned value
//   equal to zero or is the save as the previous, the `frameImg` remains
//   unchanged and the caller should retry to get the next frame.
int64_t CuvidImpl::GetFrame(cv::cuda::GpuMat &frameImg, std::string *pJpegData) {
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
		nFrameCnt = m_nCursor;
		m_Mutex.unlock();
		if (m_bBlocking) {
			m_Worker = std::thread(&CuvidImpl::__WorkerProc, this);
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

void CuvidImpl::__WorkerProc() {
	const int64_t nUserTimeScale = 1000;
	AVPacket packet;
	for (; m_Status == STATUS::WORKING; ) {
		int64_t nDecodedFrames = 0;
		if (m_nCursor == m_nFrameCnt) {
			if (m_bEOF) {
				m_Status = STATUS::STANDBY; // streaming stopped  spontaneously
				break;
			}
			av_init_packet(&packet);
			int64_t nRet = -1;
			for (; ; ::av_packet_unref(&packet)) {
				nRet = ::av_read_frame(m_pAVCtx.get(), &packet);
				if (nRet < 0 || packet.stream_index == m_nStreamId) {
					break;
				}
			}
			if (nRet >= 0 || nRet == AVERROR_EOF) {
				if (nRet == AVERROR_EOF) {
					CHECK_EQ(av_bsf_send_packet(m_pAVBsfc.get(), nullptr), 0);
				} else {
					CHECK_EQ(av_bsf_send_packet(m_pAVBsfc.get(), &packet), 0);
				}
				for (int32_t nErrCode = 0; nErrCode != AVERROR_EOF;) {
					if (m_FilterPacket.data) {
						av_packet_unref(&m_FilterPacket);
					}
					nErrCode = av_bsf_receive_packet(m_pAVBsfc.get(), &m_FilterPacket);
					if (nErrCode == AVERROR(EAGAIN)) {
						break;
					}
					uint8_t *pData = m_FilterPacket.data;
					int nSize = m_FilterPacket.size;
					int nFlag = 0;
					int64_t pts = m_FilterPacket.pts * nUserTimeScale * m_dTimeBase;
					if (nErrCode == AVERROR_EOF) {
						nFlag = CUVID_PKT_ENDOFSTREAM;
						m_bEOF = true;
					}
					nDecodedFrames += m_pDecoder->Decode(pData, nSize, nFlag, pts);
				}
			} else {
#ifdef VERBOSE_LOG
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
		}
		if (nDecodedFrames > 0 || m_nCursor < m_nFrameCnt) {
			std::lock_guard<std::mutex> locker(m_Mutex);
			m_nFrameCnt += nDecodedFrames;
			__DecodeFrame(m_WorkingBuf);
			++m_nCursor;
			if (m_bBlocking) {
				break;
			}
		}
	}
}

void CuvidImpl::SetJpegQuality(int nQuality) {
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));
	CHECK_GT(nQuality, 0);
	CHECK_LE(nQuality, 100);
	NVJPEG_CHECK(::nvjpegEncoderParamsSetQuality(m_JpegParams,
			nQuality, m_CudaStream));
}

void CuvidImpl::__DecodeFrame(cv::cuda::GpuMat &gpuImg) {
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

void CuvidImpl::__EncodeJPEG(cv::cuda::GpuMat &frameImg, std::string *pJpegData) {
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