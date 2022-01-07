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

inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id) {
    switch (id) {
    case AV_CODEC_ID_MPEG1VIDEO : return cudaVideoCodec_MPEG1;
    case AV_CODEC_ID_MPEG2VIDEO : return cudaVideoCodec_MPEG2;
    case AV_CODEC_ID_MPEG4      : return cudaVideoCodec_MPEG4;
    case AV_CODEC_ID_VC1        : return cudaVideoCodec_VC1;
    case AV_CODEC_ID_H264       : return cudaVideoCodec_H264;
    case AV_CODEC_ID_HEVC       : return cudaVideoCodec_HEVC;
    case AV_CODEC_ID_VP8        : return cudaVideoCodec_VP8;
    case AV_CODEC_ID_VP9        : return cudaVideoCodec_VP9;
    case AV_CODEC_ID_MJPEG      : return cudaVideoCodec_JPEG;
    default                     : return cudaVideoCodec_NumCodecs;
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
		, m_nCursor(0)
		, m_WorkingSema(1) {
	// CUDA Device Initialization
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));

	m_pCuCtx = ::MakeCudaContext(m_nGpuID);
	CHECK_NOTNULL(m_pCuCtx.get());
}

CuvidImpl::~CuvidImpl() {
	close();
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));
}

bool CuvidImpl::open(const std::string &strURL, READ_MODE readMode) {
	if (m_Worker.valid()) {
		auto nWaitRes = m_Worker.wait_for(std::chrono::seconds(0));
		CHECK(nWaitRes != std::future_status::timeout);
	}
	// Creating AV Context
	// -------------------
	AVDictionary *pDict = nullptr;
	CHECK_GE(::av_dict_set(&pDict, "rtsp_transport", "tcp", 0), 0);
	AVFormatContext *pAVCtxRaw = nullptr;
	auto nErrCode = ::avformat_open_input(&pAVCtxRaw, strURL.c_str(), nullptr, &pDict);
	if (nErrCode != 0) {
#ifdef VERBOSE_LOG
		char sz[1024] = {0};
		av_make_error_string(sz, 1024, nErrCode);
		LOG(WARNING) << "Can't open stream: \"" << strURL
					 << "\", err_code=" << nErrCode << ", msg=" << sz;
#endif
		return false;
	}
	m_pAVCtx.reset(pAVCtxRaw, &::DestroyAVContext);
	::av_dict_free(&pDict);

	// Find the Best Stream in AV Context
	// ----------------------------------
	AVCodec *pAVDecoder = nullptr;
	m_nStreamId = ::av_find_best_stream(m_pAVCtx.get(), AVMEDIA_TYPE_VIDEO,
			-1, -1, &pAVDecoder, 0);
	CHECK_GE(m_nStreamId, 0);
	CHECK_NOTNULL(pAVDecoder);
	AVStream *pStream = m_pAVCtx->streams[m_nStreamId];

	// Initializing AV Bit Stream Filter for H.264/H.265
	// -------------------------------------------------
	const AVBitStreamFilter *pBsf = nullptr;
	m_CurCodecId = ::FFmpeg2NvCodecId(pAVDecoder->id);
	if (m_CurCodecId == cudaVideoCodec_H264) {
		pBsf = av_bsf_get_by_name("h264_mp4toannexb");
	} else if (m_CurCodecId == cudaVideoCodec_HEVC) {
		pBsf = av_bsf_get_by_name("hevc_mp4toannexb");
	} else {
		CHECK(m_CurCodecId == cudaVideoCodec_H264 ||
			  m_CurCodecId == cudaVideoCodec_HEVC ||
			  m_CurCodecId == cudaVideoCodec_MPEG4);
	}
	if (pBsf != nullptr) {
		AVBSFContext *pBsfc = nullptr;
		CHECK_EQ(av_bsf_alloc(pBsf, &pBsfc), 0);
		m_pAVBsfc.reset(pBsfc, &::DestroyAVBsfc);
		avcodec_parameters_copy(pBsfc->par_in, pStream->codecpar);
		CHECK_EQ(av_bsf_init(pBsfc), 0);
	}

	// Initializing Internal Variables
	// -------------------------------
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

	// Creating NvDecoder with Best Stream
	// -----------------------------------
	try {
		m_pDecoder.reset(new NvDecoder(*m_pCuCtx.get(), true, m_CurCodecId, false));
	} catch (...) {
		return false;
	}

	// Streaming Started
	// -----------------
	m_nErrCode = 0;
	m_nNumDecoded = 0;
	m_nCursor = 0;
	m_FilterPacket.reset();
	m_Mp4Hdr.clear();
	m_ReadingSema.set_count(0);
	m_WorkingSema.set_count(1);
	m_Worker = std::async(std::launch::async, &CuvidImpl::__WorkerProc, this);
	return true;
}

// close streaming and clear the FAILED status
void CuvidImpl::close() {
	m_nErrCode = AVERROR_EXIT;
	m_WorkingSema.unlock();
	if (m_Worker.valid()) {
		auto waitRes = m_Worker.wait_for(std::chrono::seconds(10));
		CHECK_NE((int)waitRes, (int)std::future_status::timeout);
	}
	m_nErrCode = AVERROR_EXIT;
}

double CuvidImpl::get(cv::VideoCaptureProperties prop) const {
	CHECK_NOTNULL(m_pAVCtx);
	AVStream *pStream = m_pAVCtx->streams[m_nStreamId];
	if (prop == cv::CAP_PROP_FPS) {
		return av_q2d(pStream->avg_frame_rate);
	} else if (prop == cv::CAP_PROP_FRAME_COUNT) {
		return pStream->nb_frames;
	} else if (prop == cv::CAP_PROP_FRAME_WIDTH) {
		return pStream->codecpar->width;
	} else if (prop == cv::CAP_PROP_FRAME_HEIGHT) {
		return pStream->codecpar->height;
	}
	LOG(FATAL) << "Unsupported property: " << (int)prop;
	return 0.f;
}

int32_t CuvidImpl::errcode() const {
	return m_nErrCode;
}

// Return Value: greater or equel to zero if it working normally,
//   otherwise it is failure. Please note that if the returned value
//   equal to zero or is the save as the previous, the `frameImg` remains
//   unchanged and the caller should retry to get the next frame.
std::pair<int64_t, int64_t> CuvidImpl::read(cv::cuda::GpuMat &frameImg, uint32_t nTimeoutUS) {
	CUDA_CHECK(cudaSetDevice(m_nGpuID));
	int64_t nCursor = -1;
	int64_t nTimeStamp = -1;
	if (m_nErrCode != AVERROR_EXIT) {
		if (m_bBlocking) {
			m_ReadingSema.lock();
			if (m_nErrCode != 0) {
				if (nTimeoutUS > 0) {
					auto status = m_Worker.wait_for(std::chrono::milliseconds(nTimeoutUS));
					if (status != std::future_status::ready) {
						return std::make_pair(-1, -1);
					}
				} else {
					m_Worker.wait();
				}
				CHECK_EQ(m_nErrCode, AVERROR_EOF);
				return std::make_pair(-1, -1);
			}

			frameImg.swap(m_WorkingBuf);
			nCursor = m_nCursor;
			nTimeStamp = m_nTimeStamp;
			if (m_nErrCode == AVERROR_EOF) {
				m_nErrCode = AVERROR_EXIT;
			}
			m_WorkingSema.unlock();
		} else {
			m_ReadingSema.lock();
			std::lock_guard<std::mutex> locker(m_ReadingMutex);
			if (m_nErrCode == AVERROR_EXIT) {
				return std::make_pair(-1, -1);
			} else if (m_nErrCode != 0) {
				m_Worker.wait();
				CHECK_EQ(m_nErrCode, AVERROR_EOF);
			}
			if (!m_WorkingBuf.empty()) {
				frameImg.swap(m_WorkingBuf);
			}
			nCursor = m_nCursor;
			if (m_nErrCode == AVERROR_EOF) {
				m_nErrCode = AVERROR_EXIT;
			}
		}
	}
	m_nLastCursor = nCursor;
	return std::make_pair(nCursor, nTimeStamp);
}

void CuvidImpl::__WorkerProc() {
	CUDA_CHECK(cudaSetDevice(m_nGpuID));
	PACKET packet;
	try {
		bool bEof = false;
		for (; m_nErrCode == 0; ) {
			if (m_nNumDecoded > m_nCursor) {
				if (m_bBlocking) {
					m_WorkingSema.lock();
					m_nTimeStamp = __DecodeFrame(m_WorkingBuf);
					++m_nCursor;
					// Synchronize for preventing duplicated frames are generated
					cudaDeviceSynchronize();
					m_ReadingSema.unlock();
				} else {
					std::lock_guard<std::mutex> locker(m_ReadingMutex);
					m_nTimeStamp = __DecodeFrame(m_WorkingBuf);
					++m_nCursor;
					m_ReadingSema.unlock();
				}
			} else {
				if (bEof) {
					throw int32_t(AVERROR_EOF);
				}
				int32_t nErrCode = 0;
				for (packet.reset(); ; packet.reset()) {
					nErrCode = ::av_read_frame(m_pAVCtx.get(), packet);
					if (nErrCode < 0 || packet.get().stream_index == m_nStreamId) {
						break;
					}
				}
				if (nErrCode < 0 && nErrCode != AVERROR_EOF) {
#ifdef VERBOSE_LOG
					std::string strMsg(1024, '\0');
					auto nErr = av_strerror(nErrCode, (char *)strMsg.data(), strMsg.size());
					if (nErr < 0) {
						strMsg = "UNKNOWN";
					}
					LOG(INFO) << "One frame lost, code=" << nErrCode
								<< ", message=\"" << strMsg << "\"";
#endif
					throw nErrCode; // for catching the nErrCode
				}
				bEof = (nErrCode == AVERROR_EOF);
				if (m_CurCodecId == cudaVideoCodec_H264 ||
					m_CurCodecId == cudaVideoCodec_HEVC) {
					__DemuxH26X(packet.get(), bEof);
				} else {
					__DemuxMPG4(packet.get(), bEof);
				}
				packet.reset();
			}
		}
	} catch (int32_t nErrCode) {
		m_nErrCode = nErrCode;
	} catch (...) {
		m_nErrCode = AVERROR(EINTR);
	}
	m_ReadingSema.unlock();
}

void CuvidImpl::__DemuxH26X(AVPacket &packet, bool &bEoF) {
	int32_t nErrCode = 0;
	CHECK_NOTNULL(m_pAVBsfc);
	if (bEoF) {
		nErrCode = av_bsf_send_packet(m_pAVBsfc.get(), nullptr);
	} else {
		nErrCode = av_bsf_send_packet(m_pAVBsfc.get(), &packet);
	}
	CHECK_EQ(nErrCode, 0);
	for (bEoF = false; !bEoF;) {
		m_FilterPacket.reset();
		nErrCode = av_bsf_receive_packet(m_pAVBsfc.get(), m_FilterPacket);
		if (nErrCode == AVERROR(EAGAIN)) {
			break;
		} else if (nErrCode == AVERROR_EOF) {
			bEoF = CUVID_PKT_ENDOFSTREAM;
			nErrCode = 0;
		} else if (nErrCode != 0) {
			throw nErrCode;
		}
		m_nNumDecoded += (uint32_t)m_pDecoder->Decode(
			m_FilterPacket.get().data, m_FilterPacket.get().size, (int)bEoF,
			m_FilterPacket.get().pts * 1000 * m_dTimeBase);
	}
}

void CuvidImpl::__DemuxMPG4(AVPacket &packet, bool &bEoF) {
	m_Mp4Hdr.clear();
	if (m_nNumDecoded == 0) {
		auto nExtraSize = m_pAVCtx->streams[m_nStreamId]->codecpar->extradata_size;
		if (nExtraSize > 0) {
			m_Mp4Hdr.resize(nExtraSize + packet.size - 3);
			auto src = m_pAVCtx->streams[m_nStreamId]->codecpar->extradata;
			m_Mp4Hdr.assign(src, src + nExtraSize);
			m_Mp4Hdr.insert(m_Mp4Hdr.end(), packet.data + 3, packet.data + packet.size - 3);
		}
	}
	if (m_Mp4Hdr.empty()) {
		m_Mp4Hdr.assign(packet.data, packet.data + packet.size);
	}

	m_nNumDecoded += (uint32_t)m_pDecoder->Decode(m_Mp4Hdr.data(),
		m_Mp4Hdr.size(), (int)bEoF,
		m_FilterPacket.get().pts * 1000 * m_dTimeBase);
}

int64_t CuvidImpl::__DecodeFrame(cv::cuda::GpuMat &gpuImg) {
	CUDA_CHECK(::cudaSetDevice(m_nGpuID));

	auto frameFormat = m_pDecoder->GetOutputFormat();
	cv::Size imgSize(m_pDecoder->GetWidth(), m_pDecoder->GetHeight());
	size_t nPitch = imgSize.width * 4;
	size_t nImgBytes = imgSize.height * nPitch;
	uint8_t iMatrix = m_pDecoder->GetVideoFormatInfo()
			.video_signal_description.matrix_coefficients;
	int64_t nTimeStamp;
	uint8_t *pSrc = m_pDecoder->GetFrame(&nTimeStamp);
	CHECK_NOTNULL(pSrc);

	if (m_BgraBuf.size() != imgSize || m_BgraBuf.channels() != 4
			|| m_BgraBuf.type() != CV_8UC4) {
		m_BgraBuf = cv::cuda::GpuMat(imgSize, CV_8UC4);
	}
	uint8_t *pRGBATmp = m_BgraBuf.ptr<uint8_t>();
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
	::BGRA32ToBgr24(m_BgraBuf.data, gpuImg.data, imgSize.width, imgSize.height,
			gpuImg.step);
	return nTimeStamp;
}
