#include "prand_impl.hpp"
#include "../include/prand.hpp"

Prand::Prand(int nGpuID) {
	m_pImpl.reset(new PrandImpl(nGpuID));
}

Prand::~Prand() {
}

bool Prand::Start(const std::string &strURL, READ_MODE readMode) {
	return m_pImpl->Start(strURL, PrandImpl::READ_MODE(readMode));
}

void Prand::Stop() {
	m_pImpl->Stop();
}

double Prand::get(cv::VideoCaptureProperties prop) const {
	return m_pImpl->get(prop);
}

Prand::STATUS Prand::GetCurrentStatus() const {
	return (Prand::STATUS)m_pImpl->GetCurrentStatus();
}

int64_t Prand::GetFrame(cv::cuda::GpuMat &frameImg, std::string *pJpegData) {
	return m_pImpl->GetFrame(frameImg, pJpegData);
}

void Prand::SetJpegQuality(int nQuality) {
	m_pImpl->SetJpegQuality(nQuality);
}
