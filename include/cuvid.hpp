#ifndef __CUVID_HPP
#define __CUVID_HPP

#include "gpubuf.hpp"
#include <memory>
#include <string>
#include <utility>


class CuvidImpl;
class Cuvid {
private:
	GpuBuffer m_DefBuf;
	std::unique_ptr<CuvidImpl> m_pImpl;

public:
	enum class STATUS { STANDBY = 0, WORKING = 1, FAILED = 2 };
	enum class READ_MODE { AUTO = 0, BLOCK = 1, ASYNC = 2 };

	Cuvid(int nGpuID);
	~Cuvid();

	bool open(const std::string &strURL, READ_MODE readMode = READ_MODE::AUTO);
	void close();
	double get(int nProp) const;
	int32_t errcode() const;
	std::pair<int64_t, int64_t> read(GpuBuffer &frameImg, uint32_t nTimeoutUS = 0);
	std::pair<int64_t, int64_t> read(uint32_t nTimeoutUS = 0);

	constexpr const GpuBuffer& GetDefaultBuffer() const {
		return m_DefBuf;
	}

private:
	Cuvid(const GpuBuffer &other) = delete;
	Cuvid(GpuBuffer &&other) = delete;
	Cuvid& operator=(const Cuvid &other) = delete;
	Cuvid& operator=(Cuvid &&other) = delete;
};

#endif //__CUVID_HPP