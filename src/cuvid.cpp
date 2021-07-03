#include "cuvid_impl.hpp"
#include "../include/cuvid.hpp"

Cuvid::Cuvid(int nGpuID) {
	m_pImpl.reset(new CuvidImpl(nGpuID));
}

Cuvid::~Cuvid() {
}

bool Cuvid::Start(const std::string &strURL, READ_MODE readMode) {
	return m_pImpl->Start(strURL, CuvidImpl::READ_MODE(readMode));
}

void Cuvid::Stop() {
	m_pImpl->Stop();
}

double Cuvid::get(cv::VideoCaptureProperties prop) const {
	return m_pImpl->get(prop);
}

Cuvid::STATUS Cuvid::GetCurrentStatus() const {
	return (Cuvid::STATUS)m_pImpl->GetCurrentStatus();
}

int64_t Cuvid::GetFrame(cv::cuda::GpuMat &frameImg, std::string *pJpegData) {
	return m_pImpl->GetFrame(frameImg, pJpegData);
}

void Cuvid::SetJpegQuality(int nQuality) {
	m_pImpl->SetJpegQuality(nQuality);
}
