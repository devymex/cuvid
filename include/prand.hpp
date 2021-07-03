#ifndef __PRAND_HPP
#define __PRAND_HPP

#include <string>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

class PrandImpl;
class Prand {
public:
	enum class STATUS { STANDBY = 0, WORKING = 1, FAILED = 2 };
	enum class READ_MODE { AUTO = 0, BLOCK = 1, ASYNC = 2 };

	Prand(int nGpuID);
	~Prand();

	bool Start(const std::string &strURL, READ_MODE readMode = READ_MODE::AUTO);

	void Stop();

	double get(cv::VideoCaptureProperties prop) const;

	STATUS GetCurrentStatus() const;

	int64_t GetFrame(cv::cuda::GpuMat &frameImg,
			std::string *pJpegData = nullptr);

	void SetJpegQuality(int nQuality);

private:
	std::unique_ptr<PrandImpl> m_pImpl;
};

#endif //__PRAND_HPP