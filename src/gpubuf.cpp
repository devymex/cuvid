#include "../include/gpubuf.hpp"
#include "logging.hpp"
#include <cuda_runtime.h>

GpuBuffer::GpuBuffer()
		: m_ptr(nullptr), m_nSize(0), m_bOwner(false), m_nDevId(-1) {
}

GpuBuffer::GpuBuffer(size_t nSize)
		: m_ptr(nullptr), m_nSize(0), m_bOwner(false), m_nDevId(-1) {
	realloc(nSize);
}

GpuBuffer::GpuBuffer(void *ptr, size_t nSize, int nDevId)
		: m_ptr(ptr), m_nSize(nSize), m_bOwner(false), m_nDevId(nDevId) {
	if (m_ptr == nullptr || m_nSize == 0) {
		CHECK(m_ptr == nullptr);
		CHECK_EQ(m_nSize, 0);
		CHECK_EQ(m_nDevId, -1);
	}
	if (m_nDevId < 0) {
		CHECK_EQ(cudaSuccess, ::cudaGetDevice(&m_nDevId));
	}
}

GpuBuffer::~GpuBuffer() {
	clear();
}

void GpuBuffer::copy_to(GpuBuffer &other) const {
	if (m_nSize > 0) {
		if (m_nSize != other.size()) {
			other.realloc(m_nSize);
		}
		CHECK_EQ(cudaSuccess, ::cudaMemcpy(other.get(), m_ptr, m_nSize, cudaMemcpyDeviceToDevice));
	}
}

void GpuBuffer::to_vector(std::vector<uint8_t> &vec) const {
	vec.resize(m_nSize);
	if (m_nSize > 0) {
		CHECK_EQ(cudaSuccess, ::cudaMemcpy(vec.data(), m_ptr, m_nSize, cudaMemcpyDeviceToHost));
	}
}

void GpuBuffer::realloc(size_t nSize, int nDevId) {
	clear();
	if (nSize > 0) {
		CHECK_EQ(cudaSuccess, ::cudaMalloc(&m_ptr, nSize));
		CHECK_EQ(cudaSuccess, ::cudaGetDevice(&m_nDevId));
		m_nSize = nSize;
		m_bOwner = true;
	}
}

void GpuBuffer::clear() {
	if (m_bOwner && m_ptr) {
		CHECK_EQ(cudaSuccess, ::cudaFree(m_ptr));
	}
	m_ptr = nullptr;
	m_nSize = 0;
	m_bOwner = false;
	m_nDevId = -1;
}

void GpuBuffer::swap(GpuBuffer& other) {
	std::swap(m_ptr, other.m_ptr);
	std::swap(m_nSize, other.m_nSize);
	std::swap(m_bOwner, other.m_bOwner);
	std::swap(m_nDevId, other.m_nDevId);
}
