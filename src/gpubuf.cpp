#include "../include/gpubuf.hpp"
#include "logging.hpp"
#include <cuda_runtime.h>

GpuBuffer::GpuBuffer()
	: m_ptr(nullptr), m_nSize(0), m_bOwn(false) {
}

GpuBuffer::GpuBuffer(size_t nSize)
	: m_ptr(nullptr), m_nSize(0), m_bOwn(false) {
	if (nSize > 0) {
		CHECK_EQ(cudaSuccess, ::cudaMalloc(&m_ptr, nSize));
		m_nSize = nSize;
		m_bOwn = true;
	}
}

GpuBuffer::GpuBuffer(void *ptr, size_t nSize)
	: m_ptr(ptr), m_nSize(nSize), m_bOwn(false) {
}

GpuBuffer::~GpuBuffer() {
	if (m_bOwn && m_ptr) {
		CHECK_EQ(cudaSuccess, ::cudaFree(m_ptr));
	}
}

void* GpuBuffer::get() {
	return m_ptr;
}

void GpuBuffer::copy_to(GpuBuffer &other) const {
	if (m_nSize > 0) {
		other.resize(m_nSize);
		CHECK_EQ(cudaSuccess, ::cudaMemcpy(other.get(), m_ptr, m_nSize, cudaMemcpyDeviceToDevice));
	}
}

void GpuBuffer::to_vector(std::vector<uint8_t> &vec) const {
	vec.resize(m_nSize);
	if (m_nSize > 0) {
		CHECK_EQ(cudaSuccess, ::cudaMemcpy(vec.data(), m_ptr, m_nSize, cudaMemcpyDeviceToHost));
	}
}

void GpuBuffer::resize(size_t nSize) {
	if (nSize != m_nSize) {
		if (m_bOwn && m_ptr) {
			CHECK_EQ(cudaSuccess, ::cudaFree(m_ptr));
			m_ptr = nullptr;
		}
		CHECK_EQ(cudaSuccess, ::cudaMalloc(&m_ptr, nSize));
		m_nSize = nSize;
		m_bOwn = true;
	}
}

void GpuBuffer::swap(GpuBuffer& other) {
	std::swap(m_ptr, other.m_ptr);
	std::swap(m_nSize, other.m_nSize);
	std::swap(m_bOwn, other.m_bOwn);
}
