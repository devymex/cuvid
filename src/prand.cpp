#include <glog/logging.h>

extern "C" {
#include <libavutil/pixdesc.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>
}

#include "NvDecoder/ColorSpace.h"
#include "prand.hpp"

inline cudaVideoCodec FFmpeg2NvCodecId(AVCodecID id) {
	switch (id) {
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

Prand::Prand(std::string strURL, int nGpuID) {
	av_register_all();
	CHECK_GE(avformat_network_init(), 0);

	m_pAVCtx = avformat_alloc_context();
	CHECK_NOTNULL(m_pAVCtx);

	AVDictionary *dict = nullptr;
	CHECK_GE(av_dict_set(&dict, "rtsp_transport", "tcp", 0), 0);
	CHECK_GE(avformat_open_input(&m_pAVCtx, strURL.c_str(), nullptr, &dict), 0) << strURL;
	CHECK_GE(avformat_find_stream_info(m_pAVCtx, nullptr), 0);

	AVCodec *pAVDecoder;
	int nBestStream = av_find_best_stream(m_pAVCtx, AVMEDIA_TYPE_VIDEO, -1, -1, &pAVDecoder, 0);
	CHECK_GE(nBestStream, 0);
	// AVStream *pStream = m_pAVCtx->streams[nBestStream];
	// LOG(INFO) << pStream->codec->width;
	// LOG(INFO) << pStream->codec->height;
	cudaVideoCodec codecID = FFmpeg2NvCodecId(pAVDecoder->id);
	//LOG(INFO) << codecID;

	m_pCudaDev.reset(new CudaDevice(0));
	m_pDecoder.reset(new NvDecoder(m_pCudaDev->getContext(), true, codecID, false));
}

Prand::~Prand() {
	Stop();
	avformat_close_input(&m_pAVCtx);
}

void Prand::__DecodeFrame(const AVPacket &packet, cv::cuda::GpuMat &gpuImg) {
	auto frameFormat = m_pDecoder->GetOutputFormat();
	cv::Size imgSize(m_pDecoder->GetWidth(), m_pDecoder->GetHeight());
	size_t nPitch = imgSize.width * 4;
	size_t nImgBytes = imgSize.height * nPitch;
	uint8_t iMatrix = m_pDecoder->GetVideoFormatInfo()
			.video_signal_description.matrix_coefficients;
	uint8_t *pSrc = m_pDecoder->GetFrame();
	CHECK_NOTNULL(pSrc);

	if (gpuImg.size() != imgSize || gpuImg.type() != CV_8UC4) {
		gpuImg = cv::cuda::GpuMat(imgSize, CV_8UC4);
	}
	uint8_t *pGpuImgBuf = gpuImg.ptr<uint8_t>();
	if (m_pDecoder->GetBitDepth() == 8) {
		if (frameFormat == cudaVideoSurfaceFormat_YUV444)
			YUV444ToColor32<BGRA32>(pSrc, imgSize.width, pGpuImgBuf,
					nPitch, imgSize.width, imgSize.height, iMatrix);
		else // default assumed NV12
			Nv12ToColor32<BGRA32>(pSrc, imgSize.width, pGpuImgBuf,
					nPitch, imgSize.width, imgSize.height, iMatrix);
	} else {
		if (frameFormat == cudaVideoSurfaceFormat_YUV444)
			YUV444P16ToColor32<BGRA32>(pSrc, 2 * imgSize.width, pGpuImgBuf,
					nPitch, imgSize.width, imgSize.height, iMatrix);
		else // default assumed P016
			P016ToColor32<BGRA32>(pSrc, 2 * imgSize.width, pGpuImgBuf,
					nPitch, imgSize.width, imgSize.height, iMatrix);
	}
}

void Prand::Start() {
	m_bWorking = true;
	m_Worker = std::thread([&] {
		LOG(INFO) << "Start decoding...";
		for ( ; m_bWorking ; ) {
			AVPacket packet;
			int64_t ret = av_read_frame(m_pAVCtx, &packet);
			if (ret >= 0) {
				int64_t nDecFrames = m_pDecoder->Decode(packet.data, packet.size);
				if (nDecFrames) {
					std::lock_guard<std::mutex> locker(m_Mutex);
					__DecodeFrame(packet, m_WorkingBuf);
					m_nFrameCnt += nDecFrames;
				}
			} else {
				LOG(WARNING) << "Lost frame: " << ret;
			}
			av_packet_unref(&packet);
		}
	});
}

void Prand::Stop() {
	m_bWorking = false;
	if (m_Worker.joinable()) {
		m_Worker.join();
	}
}

int64_t Prand::GetFrame(cv::cuda::GpuMat &frameImg) {
	std::lock_guard<std::mutex> locker(m_Mutex);
	if (!m_WorkingBuf.empty()) {
		m_WorkingBuf.copyTo(frameImg);
	}
	return m_nFrameCnt;
}

int64_t Prand::GetFrame(cv::Mat &frameImg) {
	std::lock_guard<std::mutex> locker(m_Mutex);
	if (!m_WorkingBuf.empty()) {
		m_WorkingBuf.download(frameImg);
	}
	return m_nFrameCnt;
}
