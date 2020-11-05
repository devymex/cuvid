#ifndef __PRAND_HPP
#define __PRAND_HPP

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include "NvDecoder/cudevice.hpp"

class Prand { // Python RTSP AV Nvidia Decoder
public:
	Prand(std::string strURL, int nGpuID);
	~Prand();

	void Start();
	void Stop();
	int64_t GetFrame(cv::cuda::GpuMat &frameImg);
	int64_t GetFrame(cv::Mat &frameImg);

private:
	void __DecodeFrame(const AVPacket &packet, cv::cuda::GpuMat &gpuImg);

private:
	AVFormatContext *m_pAVCtx = nullptr;
	int64_t m_nFrameCnt = 0;

	std::unique_ptr<CudaDevice> m_pCudaDev;
	std::unique_ptr<NvDecoder> m_pDecoder;
	
	cv::cuda::GpuMat m_WorkingBuf;

	std::thread m_Worker;
	std::mutex m_Mutex;
	std::atomic<bool> m_bWorking;
};

#endif //__PRAND_HPP