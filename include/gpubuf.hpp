#ifndef __GPU_BUF_HPP__
#define __GPU_BUF_HPP__

#include <vector>
#include <stdint.h>
#include <stddef.h>

class GpuBuffer {
private:
	void *m_ptr;
	size_t m_nSize;
	bool m_bOwn;

public:
	GpuBuffer();
	GpuBuffer(size_t nSize);
	GpuBuffer(void *ptr, size_t nSize);
	~GpuBuffer();

	void copy_to(GpuBuffer &other) const;
	void to_vector(std::vector<uint8_t> &vec) const;
	void resize(size_t nSize);
	void swap(GpuBuffer& other);

	void* get();
	constexpr const void* get() const {
		return m_ptr;
	}
	constexpr size_t size() const {
		return m_nSize;
	}
	constexpr bool empty() const {
		return m_ptr == nullptr;
	}
	constexpr bool owns_memory() const {
		return m_bOwn;
	}

private:
	GpuBuffer(const GpuBuffer &other) = delete;
	GpuBuffer(GpuBuffer &&other) = delete;
	GpuBuffer& operator=(const GpuBuffer &other) = delete;
	GpuBuffer& operator=(GpuBuffer &&other) = delete;
};

#endif // __GPU_BUF_HPP__