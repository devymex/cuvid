#ifndef __CUVID_HPP
#define __CUVID_HPP

#include <string>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

class CuvidImpl;
class Cuvid {
public:
	enum class STATUS { STANDBY = 0, WORKING = 1, FAILED = 2 };
	enum class READ_MODE { AUTO = 0, BLOCK = 1, ASYNC = 2 };

	Cuvid(int nGpuID);
	~Cuvid();

	bool Start(const std::string &strURL, READ_MODE readMode = READ_MODE::AUTO);

	void Stop();

	double get(cv::VideoCaptureProperties prop) const;

	STATUS GetCurrentStatus() const;

	int64_t GetFrame(cv::cuda::GpuMat &frameImg,
			std::string *pJpegData = nullptr);

	void SetJpegQuality(int nQuality);

private:
	std::unique_ptr<CuvidImpl> m_pImpl;
};

#endif //__CUVID_HPP