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
	auto ret = m_pImpl->read(m_InnerBuf, nTimeoutUS);
	if (ret.first != -1) {
		if (frameImg.device_id() != m_InnerBuf.device_id() ||
				frameImg.size() != m_InnerBuf.size()) {
			m_InnerBuf.copy_to(frameImg);
		} else {
			m_InnerBuf.swap(frameImg);
		}
	}
	return ret;
}

std::pair<int64_t, int64_t> Cuvid::read(GpuBuffer &&frameImg, uint32_t nTimeoutUS) {
	auto ret = m_pImpl->read(m_InnerBuf, nTimeoutUS);
	if (ret.first != -1) {
		CHECK_EQ(frameImg.device_id(), m_InnerBuf.device_id());
		CHECK_EQ(frameImg.size(), m_InnerBuf.size());
		m_InnerBuf.copy_to(frameImg);
	}
	return ret;
}