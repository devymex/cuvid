#ifndef __GPU_BUF_HPP__
#define __GPU_BUF_HPP__

#include <vector>
#include <stdint.h>
#include <stddef.h>

class GpuBuffer {
private:
	void *m_ptr;
	size_t m_nSize;
	bool m_bOwner;
	int m_nDevId;

public:
	GpuBuffer();
	GpuBuffer(size_t nSize);
	GpuBuffer(void *ptr, size_t nSize, int nDevId = -1);
	~GpuBuffer();

	void realloc(size_t nSize, int nDevId = -1);
	void copy_to(GpuBuffer &other) const;
	void to_vector(std::vector<uint8_t> &vec) const;
	void swap(GpuBuffer& other);
	void clear();

	constexpr void* get() {
		return m_ptr;
	}
	constexpr const void* get() const {
		return m_ptr;
	}
	constexpr size_t size() const {
		return m_nSize;
	}
	constexpr bool empty() const {
		return m_ptr == nullptr;
	}
	constexpr bool is_owner() const {
		return m_bOwner;
	}
	constexpr int device_id() const {
		return m_nDevId;
	}

private:
	GpuBuffer(const GpuBuffer &other) = delete;
	GpuBuffer(GpuBuffer &&other) = delete;
	GpuBuffer& operator=(const GpuBuffer &other) = delete;
	GpuBuffer& operator=(GpuBuffer &&other) = delete;
};

#endif // __GPU_BUF_HPP__