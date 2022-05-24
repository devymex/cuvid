#include "../include/cuvid.hpp"
#include "../src/logging.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <Python.h>

extern "C" {

PyObject* NDArrayFromData(const std::vector<long> &shape, uint8_t *pData) {
	PyObject *pTmp = PyArray_SimpleNewFromData(shape.size(),
			shape.data(), NPY_UBYTE, pData);
	PyObject *pRet = PyArray_NewCopy((PyArrayObject*)pTmp, NPY_ANYORDER);
	Py_XDECREF(pTmp);
	return pRet;
}

void CuvidDestroy(PyObject *pArgs) {
#ifdef VERBOSE_LOG
	LOG(INFO) << "Cuvid Destructed";
#endif
	delete (Cuvid*)PyCapsule_GetPointer(pArgs, "Cuvid");
}

PyObject* CuvidCreate(PyObject *self, PyObject *pArgs) {
	int nDevID = 0;
	CHECK(PyArg_ParseTuple(pArgs, "i", &nDevID));

#ifdef VERBOSE_LOG
	LOG(INFO) << "Cuvid Created, DevID=" << nDevID;
#endif

	auto pCuvid = new Cuvid(nDevID);
	return PyCapsule_New((void*)pCuvid, "Cuvid", CuvidDestroy);
}

PyObject* CuvidOpen(PyObject *self, PyObject *pArgs) {
	PyObject *pCapsule;
	char *pURL = nullptr;
	auto nReadMode = (int32_t)Cuvid::READ_MODE::AUTO;
	CHECK(PyArg_ParseTuple(pArgs, "Os|i", &pCapsule, &pURL, &nReadMode));
	auto pCuvid = (Cuvid*)PyCapsule_GetPointer(pCapsule, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	CHECK_GE(nReadMode, 0);
	CHECK_LE(nReadMode, 2);

	PyObject *pyResult = Py_False;
	if (pCuvid->open(pURL, Cuvid::READ_MODE(nReadMode))) {
		Py_RETURN_TRUE;
	}
	Py_RETURN_FALSE;
}

PyObject* CuvidClose(PyObject *self, PyObject *pArgs) {
	auto pCuvid = (Cuvid*)PyCapsule_GetPointer(pArgs, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	pCuvid->close();
	Py_RETURN_NONE;
}

PyObject* CuvidGet(PyObject *self, PyObject *pArgs) {
	PyObject *pCapsule;
	int32_t nProp = -1;
	CHECK(PyArg_ParseTuple(pArgs, "Oi", &pCapsule, &nProp));

	auto pCuvid = (Cuvid*)PyCapsule_GetPointer(pCapsule, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	auto dVal = pCuvid->get(nProp);
	return PyFloat_FromDouble(dVal);
}

PyObject* CuvidErrCode(PyObject *self, PyObject *pArgs) {
	auto pCuvid = (Cuvid*)PyCapsule_GetPointer(pArgs, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	return PyLong_FromLong(pCuvid->errcode());
}

PyObject* CuvidReadAsNumpy(PyObject *self, PyObject *pArgs) {
	auto pCuvid = (Cuvid*)PyCapsule_GetPointer(pArgs, "Cuvid");
	CHECK_NOTNULL(pCuvid);

	auto [nFrameCnt, nTimeStamp] = pCuvid->read();
	auto &gpuBuf = pCuvid->GetDefaultBuffer();
	std::vector<uint8_t> cpuBuf;
	gpuBuf.to_vector(cpuBuf);

	PyObject *pNpImg = nullptr, *pRet = nullptr;
	if (!cpuBuf.empty()) {
		std::vector<long> shape = { (long)pCuvid->get(4),
				(long)pCuvid->get(3), (long)pCuvid->get(6) };
		pNpImg = NDArrayFromData(shape, cpuBuf.data());
	} else {
		pNpImg = Py_None;
		Py_XINCREF(pNpImg);
	}
	pRet = PyTuple_Pack(3, PyLong_FromLong(nFrameCnt),
			PyLong_FromLong(nTimeStamp), pNpImg);
	Py_XDECREF(pNpImg);
	return pRet;
}

PyObject* CuvidReadToBuffer(PyObject *self, PyObject *pArgs) {
	PyObject *pCapsule;
	int64_t nBufPtr;
	int64_t nBufSize;
	CHECK(PyArg_ParseTuple(pArgs, "Oll", &pCapsule, &nBufPtr, &nBufSize));

	auto pCuvid = (Cuvid*)PyCapsule_GetPointer(pCapsule, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	CHECK_EQ(nBufSize, (int32_t)pCuvid->get(8)) << "Expeted buffer size "
			<< pCuvid->get(8) << " but got " << nBufSize;

	unsigned long nPtr64 = nBufPtr;
	GpuBuffer gpuBuf((void*)nPtr64, nBufSize);
	auto [nFrameCnt, nTimeStamp] = pCuvid->read();
	pCuvid->GetDefaultBuffer().copy_to(gpuBuf);
	auto pRet = PyTuple_Pack(2, PyLong_FromLong(nFrameCnt),
			PyLong_FromLong(nTimeStamp));
	return pRet;
}

static PyMethodDef cuvid_methods[] = {
	{
		"cuvid_create", (PyCFunction)CuvidCreate,
		METH_VARARGS, "Create a cuvid object."
	}, {
		"cuvid_open", (PyCFunction)CuvidOpen,
		METH_VARARGS, "Please make sure current status is STANDBY."
	}, {
		"cuvid_close", (PyCFunction)CuvidClose,
		METH_O, "Stop streaming and clear fail status."
	}, {
		"cuvid_get", (PyCFunction)CuvidGet,
		METH_VARARGS, "Get information from videos."
	}, {
		"cuvid_errcode", (PyCFunction)CuvidErrCode,
		METH_O, "[Status] 0: STANDBY, 1: WORKING, -1: FAILED"
	}, {
		"cuvid_read_as_numpy", (PyCFunction)CuvidReadAsNumpy,
		METH_O, "[Return code] 0: Empty -1: Failed, >0: Successed"
	}, {
		"cuvid_read_to_buffer", (PyCFunction)CuvidReadToBuffer,
		METH_VARARGS, "[Return code] 0: Empty -1: Failed, >0: Successed"
	},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef cuvid_definition = {
	PyModuleDef_HEAD_INIT,
	"cuvid",
	"Python RTSP AV Nvidia Decoder",
	-1,
	cuvid_methods
};

PyMODINIT_FUNC PyInit_cuvid(void) {
	Py_Initialize();
	import_array();
	return PyModule_Create(&cuvid_definition);
}

}
