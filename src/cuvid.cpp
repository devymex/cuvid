#include "cuvid_impl.hpp"
#include "../include/cuvid.hpp"

Cuvid::Cuvid(int nGpuID) {
	m_pImpl.reset(new CuvidImpl(nGpuID));
}

Cuvid::~Cuvid() {
}

bool Cuvid::open(const std::string &strURL, READ_MODE readMode) {
	return m_pImpl->open(strURL, CuvidImpl::READ_MODE(readMode));
}

void Cuvid::close() {
	m_pImpl->close();
}

double Cuvid::get(cv::VideoCaptureProperties prop) const {
	return m_pImpl->get(prop);
}

int32_t Cuvid::errcode() const {
	return m_pImpl->errcode();
}

int64_t Cuvid::read(cv::cuda::GpuMat &frameImg) {
	return m_pImpl->read(frameImg);
}
