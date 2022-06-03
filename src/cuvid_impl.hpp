#ifndef __CUVID_IMPL_HPP
#define __CUVID_IMPL_HPP

#include "../include/gpubuf.hpp"
#include "semaphore.hpp"
#include "NvDecoder/NvDecoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

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

struct AVPacketUnref {
    void operator()(AVPacket *pPacket) const {
        av_packet_free(&pPacket);
    }
};

struct AVPACKET {
	std::unique_ptr<AVPacket, AVPacketUnref> m_pPacket;
	AVPACKET() {
		m_pPacket.reset(av_packet_alloc());
	}

	void reset() {
		m_pPacket.reset(av_packet_alloc());
	}

	AVPacket* operator->() {
		return m_pPacket.get();
	}

	AVPacket* get() {
		return m_pPacket.get();
	}

	AVPacket& operator*() {
		return *m_pPacket;
	}

	AVPACKET(const AVPACKET &) = delete;
	AVPACKET& operator =(const AVPACKET &) = delete;
};

// Python RTSP AV Nvidia Decoder
class CuvidImpl {
public:
	enum class READ_MODE { AUTO = 0, BLOCK = 1, ASYNC = 2 };

	CuvidImpl(int nGpuID);

	~CuvidImpl();

	bool open(const std::string &strURL, READ_MODE readMode = READ_MODE::AUTO, uint32_t nTimeoutMS = 0);

	void close();

	double get(int nProp) const;

	int32_t errcode() const;

	std::pair<int64_t, int64_t> read(GpuBuffer &frameImg, uint32_t nTimeoutUS = 0);

private:
	int64_t __DecodeFrame(GpuBuffer &gpuImg);

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

	// Video Info
	int m_nStreamId;
	cudaVideoCodec m_CurCodecId;
	double m_dTimeBase;

	// Video temp date
	std::vector<uint8_t> m_Mp4Hdr;
	AVPACKET m_FilterPacket;

	// Producer & Customer
	int64_t m_nLastCursor;
	std::atomic<int64_t> m_nCursor;
	std::atomic<int64_t> m_nTimeStamp;
	std::atomic<int64_t> m_nNumDecoded;
	std::atomic<int32_t> m_nErrCode;
	std::mutex m_ReadingMutex;
	semaphore m_WorkingSema;
	semaphore m_ReadingSema;
	std::future<void> m_Worker;
	GpuBuffer m_BgraBuf;
	GpuBuffer m_WorkingBuf;
	GpuBuffer m_ReadingBuf;
};

#endif //__CUVID_IMPL_HPP