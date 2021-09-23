#ifndef __CUVID_IMPL_HPP
#define __CUVID_IMPL_HPP

#include "semaphore.hpp"
#include "NvDecoder/NvDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <nvjpeg.h>

#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <future>

#define MAKE_STR(name) (#name)

#define CUDA_CHECK(exp) {auto e = exp; if (e != cudaSuccess) { \
		LOG(FATAL) << "CUDA call " << MAKE_STR(exp) << \
			" failed: err_code=" << e << ", err_name=\"" << \
			cudaGetErrorName(e) << "\", err_msg=\"" << \
			cudaGetErrorString(e) << "\""; \
	}}

#define NVJPEG_CHECK(exp) {auto e = exp; if (e != NVJPEG_STATUS_SUCCESS) { \
		LOG(FATAL) << "NVJPEG call " << MAKE_STR(exp) << \
			" failed: err_code=" << e; \
	}}

struct PACKET {
	AVPacket _packet;
	PACKET() {
		_packet.data = nullptr;
		_packet.size = 0;
		av_init_packet(&_packet);
	}
	~PACKET() {
		av_packet_unref(&_packet);
	}

	void reset() {
		av_packet_unref(&_packet);
		av_init_packet(&_packet);
	}

	operator AVPacket* () {
		return &_packet;
	}

	AVPacket& get() {
		return _packet;
	}

	PACKET(const PACKET &) = delete;
	PACKET& operator =(const PACKET &) = delete;
};

// Python RTSP AV Nvidia Decoder
class CuvidImpl {
public:
	enum class READ_MODE { AUTO = 0, BLOCK = 1, ASYNC = 2 };

	CuvidImpl(int nGpuID);

	~CuvidImpl();

	bool open(const std::string &strURL, READ_MODE readMode = READ_MODE::AUTO);

	void close();

	double get(cv::VideoCaptureProperties prop) const;

	int32_t errcode() const;

	int64_t read(cv::cuda::GpuMat &frameImg);

private:
	void __DecodeFrame(cv::cuda::GpuMat &gpuImg);

	void __WorkerProc();

	void __DemuxH26X(AVPacket &packet, bool &bEoF);

	void __DemuxMPG4(AVPacket &packet, bool &bEoF);

private:
	// User settings
	int m_nGpuID;
	bool m_bBlocking;

	// NV Decoder
	std::shared_ptr<CUcontext> m_pCuCtx;
	std::unique_ptr<NvDecoder> m_pDecoder;
	std::shared_ptr<AVFormatContext> m_pAVCtx;
	std::shared_ptr<AVBSFContext> m_pAVBsfc;
	cv::cuda::GpuMat m_BgraBuf;

	// Video Info
	int m_nStreamId;
	cudaVideoCodec m_CurCodecId;
	double m_dTimeBase;

	// Video temp date
	std::vector<uint8_t> m_Mp4Hdr;
	PACKET m_FilterPacket;

	// Producer & Customer
	std::atomic<uint64_t> m_nNumDecoded;
	std::atomic<int64_t> m_nCursor;
	std::atomic<int32_t> m_nErrCode;
	cv::cuda::GpuMat m_WorkingBuf;
	cv::cuda::GpuMat m_ReadingBuf;
	std::mutex m_ReadingMutex;
	semaphore m_WorkingSema;
	semaphore m_ReadingSema;
	std::future<void> m_Worker;
};

#endif //__CUVID_IMPL_HPP