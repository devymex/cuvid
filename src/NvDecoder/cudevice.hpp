#ifndef __CUDEVICE_HPP
#define __CUDEVICE_HPP

#include <memory>
#include "NvDecoder.h"

class CudaDevice {
public:
	CudaDevice(int nGpuID);
	~CudaDevice();
	CUdevice& getDevice();
	CUcontext& getContext();

private:
	std::shared_ptr<CUdevice> cuDev = nullptr;
	std::shared_ptr<CUcontext> cuCtx = nullptr;
};

#endif //__CUDEVICE_HPP
