#ifndef __PRAND_IMPL_HPP
#define __PRAND_IMPL_HPP

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

// Python RTSP AV Nvidia Decoder
class PrandImpl {
public:
	enum class STATUS { STANDBY = 0, WORKING = 1, FAILED = 2 };
	enum class READ_MODE { AUTO = 0, BLOCK = 1, ASYNC = 2 };

	PrandImpl(int nGpuID);

	~PrandImpl();

	std::pair<bool, cv::Size> Start(const std::string &strURL,
			READ_MODE readMode = READ_MODE::AUTO);

	void Stop();

	STATUS GetCurrentStatus() const;

	int64_t GetFrame(cv::cuda::GpuMat &frameImg,
			std::string *pJpegData = nullptr);

	void SetJpegQuality(int nQuality);

private:
	void __DecodeFrame(cv::cuda::GpuMat &gpuImg);

	void __WorkerProc();

	void __EncodeJPEG(cv::cuda::GpuMat &frameImg, std::string *pJpegData);

private:
	int m_nGpuID = 0;
	int m_nStreamId = -1;
	bool m_bBlocking = false;
	bool m_bEOF = false;
	double m_dTimeBase = 0.;

	std::shared_ptr<AVFormatContext> m_pAVCtx;
	std::shared_ptr<AVBSFContext> m_pAVBsfc;
	std::atomic<int64_t> m_nFrameCnt;
	std::atomic<int64_t> m_nCursor;
	AVPacket m_FilterPacket;
	cudaVideoCodec m_CurCodecId;

	std::shared_ptr<CUcontext> m_pCuCtx;
	std::unique_ptr<NvDecoder> m_pDecoder;
	
	cv::cuda::GpuMat m_WorkingBuf;
	cv::cuda::GpuMat m_BGRATmp;

	std::thread m_Worker;
	std::mutex m_Mutex;
	std::atomic<STATUS> m_Status;

	cudaStream_t m_CudaStream;
	nvjpegHandle_t m_JpegHandle;
	nvjpegEncoderState_t m_JpegState;
	nvjpegEncoderParams_t m_JpegParams;
};

#endif //__PRAND_IMPL_HPP