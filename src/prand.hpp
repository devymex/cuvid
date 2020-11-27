#ifndef __PRAND_HPP
#define __PRAND_HPP

#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <nvjpeg.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "NvDecoder/NvDecoder.h"

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
class Prand {
public:
	Prand(std::string strURL, int nGpuID);
	~Prand();

	void Start();
	void Stop();
	int64_t GetFrame(cv::cuda::GpuMat &frameImg,
			std::string *pJpegData = nullptr);
	void SetJpegQuality(int nQuality);

private:
	void __Initialize();
	void __ReinitRTSP();
	void __DecodeFrame(const AVPacket &packet, cv::cuda::GpuMat &gpuImg);
	void __WorkerProc();

private:
	std::string m_strURL;
	int m_nGpuID = 0;

	std::shared_ptr<AVFormatContext> m_pAVCtx;
	std::atomic<int64_t> m_nFrameCnt;

	std::shared_ptr<CUcontext> m_pCuCtx;
	std::unique_ptr<NvDecoder> m_pDecoder;
	
	cv::cuda::GpuMat m_WorkingBuf;
	cv::cuda::GpuMat m_BGRATmp;

	std::thread m_Worker;
	std::mutex m_Mutex;
	std::atomic<bool> m_bWorking;

	cudaStream_t m_CudaStream;
	nvjpegHandle_t m_JpegHandle;
	nvjpegEncoderState_t m_JpegState;
	nvjpegEncoderParams_t m_JpegParams;
};

#endif //__PRAND_HPP