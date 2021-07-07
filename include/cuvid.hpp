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

	bool open(const std::string &strURL, READ_MODE readMode = READ_MODE::AUTO);

	void close();

	double get(cv::VideoCaptureProperties prop) const;

	STATUS status() const;

	int64_t read(cv::cuda::GpuMat &frameImg, std::string *pJpegData = nullptr);

	void setJpegQuality(int nQuality);

private:
	std::unique_ptr<CuvidImpl> m_pImpl;
};

#endif //__CUVID_HPP