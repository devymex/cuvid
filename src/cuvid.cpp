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

double Cuvid::get(int nProp) const {
	return m_pImpl->get(nProp);
}

int32_t Cuvid::errcode() const {
	return m_pImpl->errcode();
}

std::pair<int64_t, int64_t> Cuvid::read(GpuBuffer &frameImg, uint32_t nTimeoutUS) {
	return m_pImpl->read(frameImg, nTimeoutUS);
}

std::pair<int64_t, int64_t> Cuvid::read(uint32_t nTimeoutUS) {
	return read(m_DefBuf, nTimeoutUS);
}
