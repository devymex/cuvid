#include <memory>
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
#include "prand.hpp"

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
	::avformat_close_input(&pAVCtx);
}

Prand::Prand(std::string strURL, int nGpuID)
		: m_strURL(strURL), m_nGpuID(nGpuID)
		, m_nFrameCnt(0), m_bWorking(false) {
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));
	CUDA_CHECK(::cudaStreamCreate(&m_CudaStream));
	NVJPEG_CHECK(::nvjpegCreateSimple(&m_JpegHandle));
	NVJPEG_CHECK(::nvjpegEncoderStateCreate(m_JpegHandle,
			&m_JpegState, m_CudaStream));
	NVJPEG_CHECK(::nvjpegEncoderParamsCreate(m_JpegHandle,
			&m_JpegParams, m_CudaStream));

	NVJPEG_CHECK(::nvjpegEncoderParamsSetSamplingFactors(m_JpegParams,
			NVJPEG_CSS_444, m_CudaStream));
	NVJPEG_CHECK(::nvjpegEncoderParamsSetEncoding(m_JpegParams,
			NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, m_CudaStream));
	NVJPEG_CHECK(::nvjpegEncoderParamsSetQuality(m_JpegParams,
			80, m_CudaStream));
	NVJPEG_CHECK(::nvjpegEncoderParamsSetOptimizedHuffman(m_JpegParams,
			1, m_CudaStream));
}

Prand::~Prand() {
	Stop();
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));
	NVJPEG_CHECK(::nvjpegEncoderParamsDestroy(m_JpegParams));
	NVJPEG_CHECK(::nvjpegEncoderStateDestroy(m_JpegState));
	NVJPEG_CHECK(::nvjpegDestroy(m_JpegHandle));
	CUDA_CHECK(::cudaStreamDestroy(m_CudaStream));
}

void Prand::Start() {
	if (m_pAVCtx == nullptr) {
		__Initialize();
	}
	m_bWorking = true;
	m_Worker = std::thread(&Prand::__WorkerProc, this);
}

void Prand::Stop() {
	m_bWorking = false;
	if (m_Worker.joinable()) {
		m_Worker.join();
	}
}

int64_t Prand::GetFrame(cv::cuda::GpuMat &frameImg, std::string *pJpegData) {
	std::lock_guard<std::mutex> locker(m_Mutex);
	CUDA_CHECK(cudaSetDevice(m_nGpuID));
	if (!m_WorkingBuf.empty()) {
		try {
			m_WorkingBuf.copyTo(frameImg);
		} catch (...) {
			LOG(FATAL) << "m_WorkingBuf.copyTo(frameImg)";
		}
		if (pJpegData != nullptr) {
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
	}
	return m_nFrameCnt;
}

void Prand::SetJpegQuality(int nQuality) {
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));
	CHECK_GT(nQuality, 0);
	CHECK_LE(nQuality, 100);
	NVJPEG_CHECK(::nvjpegEncoderParamsSetQuality(m_JpegParams,
			nQuality, m_CudaStream));
}

void Prand::__Initialize() {
	CHECK(m_pAVCtx.get() == nullptr);

	__ReinitRTSP();
	CHECK_GE(::avformat_find_stream_info(m_pAVCtx.get(), nullptr), 0);
	AVCodec *pAVDecoder = nullptr;
	int nBestStream = ::av_find_best_stream(m_pAVCtx.get(), AVMEDIA_TYPE_VIDEO,
			-1, -1, &pAVDecoder, 0);
	CHECK_GE(nBestStream, 0);
	CHECK_NOTNULL(pAVDecoder);
	// LOG(INFO) << codecID;
	// AVStream *pStream = m_pAVCtx->streams[nBestStream];
	// LOG(INFO) << pStream->codec->width;
	// LOG(INFO) << pStream->codec->height;

	CUDA_CHECK(::cudaSetDevice(m_nGpuID));
	m_pCuCtx = ::MakeCudaContext(m_nGpuID);
	cudaVideoCodec codecID = ::FFmpeg2NvCodecId(pAVDecoder->id);
	m_pDecoder.reset(new NvDecoder(*m_pCuCtx.get(), true, codecID, false));
}

void Prand::__ReinitRTSP() {
	AVDictionary *pDict = nullptr;
	CHECK_GE(::av_dict_set(&pDict, "rtsp_transport", "tcp", 0), 0);
	AVFormatContext *pAVCtxRaw = nullptr;
	CHECK_GE(::avformat_open_input(&pAVCtxRaw, m_strURL.c_str(),
			nullptr, &pDict), 0) << m_strURL;
	m_pAVCtx.reset(pAVCtxRaw, &::DestroyAVContext);
	::av_dict_free(&pDict);
}

void Prand::__WorkerProc() {
	for (; m_bWorking; ) {
		AVPacket packet = {0};
		int64_t nRet = ::av_read_frame(m_pAVCtx.get(), &packet);
		if (nRet >= 0) {
			int64_t nDecFrames = m_pDecoder->Decode(packet.data, packet.size);
			if (nDecFrames) {
				std::lock_guard<std::mutex> locker(m_Mutex);
				__DecodeFrame(packet, m_WorkingBuf);
				m_nFrameCnt += nDecFrames;
			}
		} else {
			LOG(WARNING) << "Lost one frame, nRet=" << nRet;
			__ReinitRTSP();
		}
		av_packet_unref(&packet);
	}
}

void Prand::__DecodeFrame(const AVPacket &packet, cv::cuda::GpuMat &gpuImg) {
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
