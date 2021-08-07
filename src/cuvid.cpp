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

Cuvid::STATUS Cuvid::status() const {
	return (Cuvid::STATUS)m_pImpl->status();
}

int64_t Cuvid::read(cv::cuda::GpuMat &frameImg, std::string *pJpegData) {
	return m_pImpl->read(frameImg, pJpegData);
}

void Cuvid::setJpegQuality(int nQuality) {
	m_pImpl->setJpegQuality(nQuality);
}