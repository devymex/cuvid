#include "../src/cuvid_impl.hpp"
#include "../src/logging.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <Python.h>

extern "C" {

void CuvidDestroy(PyObject *pCapsule) {
#ifdef VERBOSE_LOG
	LOG(INFO) << "Cuvid Destructed";
#endif
	delete (CuvidImpl*)PyCapsule_GetPointer(pCapsule, "Cuvid");
}

PyObject* CuvidCreate(PyObject *self, PyObject *pArgs) {
	int nDevID = 0;
	CHECK(PyArg_ParseTuple(pArgs, "i", &nDevID));

#ifdef VERBOSE_LOG
	LOG(INFO) << "Cuvid Created, DevID=" << nDevID;
#endif

	CuvidImpl *pCuvid = new CuvidImpl(nDevID);
	return PyCapsule_New((void*)pCuvid, "Cuvid", CuvidDestroy);
}

PyObject* CuvidOpen(PyObject *self, PyObject *pArgs) {
	PyObject *pObj;
	char *pURL = nullptr;
	auto nReadMode = (int32_t)CuvidImpl::READ_MODE::AUTO;
	CHECK(PyArg_ParseTuple(pArgs, "Os|i", &pObj, &pURL, &nReadMode));
	auto pCuvid = (CuvidImpl*)PyCapsule_GetPointer(pObj, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	CHECK_GE(nReadMode, 0);
	CHECK_LE(nReadMode, 2);

	PyObject *pyResult = Py_False;
	if (pCuvid->open(pURL, CuvidImpl::READ_MODE(nReadMode))) {
		Py_RETURN_TRUE;
	}
	Py_RETURN_FALSE;
}

PyObject* CuvidClose(PyObject *self, PyObject *pCapsule) {
	auto pCuvid = (CuvidImpl*)PyCapsule_GetPointer(pCapsule, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	pCuvid->close();
	Py_RETURN_NONE;
}

PyObject* CuvidGet(PyObject *self, PyObject *pArgs) {
	PyObject *pObj;
	int32_t nProp = -1;
	CHECK(PyArg_ParseTuple(pArgs, "Oi", &pObj, &nProp));

	auto pCuvid = (CuvidImpl*)PyCapsule_GetPointer(pObj, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	auto dVal = pCuvid->get(cv::VideoCaptureProperties(nProp));
	return PyFloat_FromDouble(dVal);
}

PyObject* CuvidErrCode(PyObject *self, PyObject *pCapsule) {
	auto pCuvid = (CuvidImpl*)PyCapsule_GetPointer(pCapsule, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	return PyLong_FromLong(pCuvid->errcode());
}

PyObject* NDArrayFromData(const std::vector<long> &shape, uint8_t *pData) {
	PyObject *pTmp = PyArray_SimpleNewFromData(shape.size(),
			shape.data(), NPY_UBYTE, pData);
	PyObject *pRet = PyArray_NewCopy((PyArrayObject*)pTmp, NPY_ANYORDER);
	Py_XDECREF(pTmp);
	return pRet;
}

PyObject* CuvidRead(PyObject *self, PyObject *pCapsule) {
	auto pCuvid = (CuvidImpl*)PyCapsule_GetPointer(pCapsule, "Cuvid");
	CHECK_NOTNULL(pCuvid);

	cv::cuda::GpuMat gpuImg;
	std::string strJpeg;
	auto [nFrameCnt, nTimeStamp] = pCuvid->read(gpuImg);

	cv::Mat img;
	if (nFrameCnt >= 0 && !gpuImg.empty()) {
		gpuImg.download(img);
	}

	PyObject *pNpImg = nullptr, *pRet = nullptr;
	if (!img.empty()) {
		std::vector<long> shape = { img.rows, img.cols, img.channels() };
		pNpImg = NDArrayFromData(shape, img.data);
	} else {
		pNpImg = Py_None;
		Py_XINCREF(pNpImg);
	}
	pRet = PyTuple_Pack(3, PyLong_FromLong(nFrameCnt),
			PyLong_FromLong(nTimeStamp), pNpImg);
	Py_XDECREF(pNpImg);
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
		"cuvid_status", (PyCFunction)CuvidErrCode,
		METH_O, "[Status] 0: STANDBY, 1: WORKING, -1: FAILED"
	}, {
		"cuvid_read", (PyCFunction)CuvidRead,
		METH_O, "[Return code] 0: Empty -1: Failed, >0: Successed"
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
