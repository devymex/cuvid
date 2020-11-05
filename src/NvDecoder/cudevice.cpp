#include "cudevice.hpp"
#include <glog/logging.h>

CudaDevice::CudaDevice(int nGpuID) {
	cuDev = std::make_shared<CUdevice>();
	cuCtx = std::make_shared<CUcontext>();

	CUDA_DRVAPI_CALL(cuInit(0));
	int nGpu = 0;
	CUDA_DRVAPI_CALL(cuDeviceGetCount(&nGpu));
	if (nGpuID < 0 || nGpuID >= nGpu) {
		LOG(INFO) << "GPU ordinal out of range. Should be within [" << 0 << ",  " << nGpu - 1 << "]" << std::endl;
	}

	CUDA_DRVAPI_CALL(cuDeviceGet(cuDev.get(), nGpuID));
	char szDeviceName[80];
	CUDA_DRVAPI_CALL(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), *cuDev.get()));
	LOG(INFO) << "GPU in use: " << szDeviceName << std::endl;

	CUDA_DRVAPI_CALL(cuCtxCreate(cuCtx.get(), 0, *cuDev.get()));
}

CudaDevice::~CudaDevice() {
	cuCtxDestroy(*cuCtx.get());
}

CUdevice& CudaDevice::getDevice() {
	return *cuDev;
}

CUcontext& CudaDevice::getContext() {
	return *cuCtx;
}
