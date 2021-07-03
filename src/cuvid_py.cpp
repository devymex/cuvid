#include "cuvid_impl.hpp"
#include "logging.hpp"

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
	LOG(INFO) << "Cuvid Created, DevID=nDevID";
#endif

	CuvidImpl *pCuvid = new CuvidImpl(nDevID);
	return PyCapsule_New((void*)pCuvid, "Cuvid", CuvidDestroy);
}

PyObject* CuvidStart(PyObject *self, PyObject *pArgs) {
	PyObject *pObj;
	char *pURL = nullptr;
	CHECK(PyArg_ParseTuple(pArgs, "Os", &pObj, &pURL));
	auto pCuvid = (CuvidImpl*)PyCapsule_GetPointer(pObj, "Cuvid");
	CHECK_NOTNULL(pCuvid);

	PyObject *pyResult = Py_False;
	if (pCuvid->Start(pURL)) {
		Py_RETURN_TRUE;
	}
	Py_RETURN_FALSE;
}

PyObject* CuvidStop(PyObject *self, PyObject *pCapsule) {
	auto pCuvid = (CuvidImpl*)PyCapsule_GetPointer(pCapsule, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	pCuvid->Stop();
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

PyObject* CuvidGetCurrentStatus(PyObject *self, PyObject *pCapsule) {
	auto pCuvid = (CuvidImpl*)PyCapsule_GetPointer(pCapsule, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	auto status = pCuvid->GetCurrentStatus();
	long nStatus = 0;
	switch (status) {
	case CuvidImpl::STATUS::FAILED: nStatus = -1; break;
	case CuvidImpl::STATUS::WORKING: nStatus = 1; break;
	}
	return PyLong_FromLong(nStatus);
}

PyObject* CuvidSetJpegQuality(PyObject *self, PyObject *pArgs) {
	PyObject *pObj;
	int nQuality = -1;
	CHECK(PyArg_ParseTuple(pArgs, "Oi", &pObj, &nQuality));

	auto pCuvid = (CuvidImpl*)PyCapsule_GetPointer(pObj, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	pCuvid->SetJpegQuality(nQuality);
	Py_RETURN_NONE;
}

PyObject* NDArrayFromData(const std::vector<long> &shape, uint8_t *pData) {
	PyObject *pTmp = PyArray_SimpleNewFromData(shape.size(),
			shape.data(), NPY_UBYTE, pData);
	PyObject *pRet = PyArray_NewCopy((PyArrayObject*)pTmp, NPY_ANYORDER);
	Py_XDECREF(pTmp);
	return pRet;
}

PyObject* CuvidGetFrame(PyObject *self, PyObject *pArgs) {
	PyObject *pObj;
	int nWithJpeg = 0;
	CHECK(PyArg_ParseTuple(pArgs, "O|i", &pObj, &nWithJpeg));

	auto pCuvid = (CuvidImpl*)PyCapsule_GetPointer(pObj, "Cuvid");
	CHECK_NOTNULL(pCuvid);
	cv::cuda::GpuMat gpuImg;
	std::string strJpeg;
	int64_t nFrameCnt = pCuvid->GetFrame(gpuImg,
			nWithJpeg > 0 ? &strJpeg : nullptr);
	cv::Mat img;
	if (nFrameCnt >= 0 && !gpuImg.empty()) {
		gpuImg.download(img);
	}

	PyObject *pNpImg = Py_None;
	PyObject *pJpeg = Py_None;
	PyObject *pRet = Py_None;
	if (!img.empty()) {
		std::vector<long> shape = { img.rows, img.cols, img.channels() };
		pNpImg = NDArrayFromData(shape, img.data);
	} else {
		Py_XINCREF(pNpImg);
	}
	if (nWithJpeg) {
		if (!strJpeg.empty()) {
#ifdef JPEG_AS_NUMPY
			pJpeg = NDArrayFromData({ (int)strJpeg.size() }, strJpeg.data());
#else // #ifdef FRAME_AS_NUMPY
			pJpeg = PyBytes_FromStringAndSize(strJpeg.data(), strJpeg.size());
#endif // #ifdef FRAME_AS_NUMPY
		} else {
			Py_XINCREF(pJpeg);
		}
		pRet = PyTuple_Pack(3, PyLong_FromLong(nFrameCnt), pNpImg, pJpeg);
	} else {
		pRet = PyTuple_Pack(2, PyLong_FromLong(nFrameCnt), pNpImg);
	}
	Py_XDECREF(pJpeg);
	Py_XDECREF(pNpImg);
	return pRet;
}

static PyMethodDef cuvid_methods[] = { 
	{
		"cuvid_create", (PyCFunction)CuvidCreate,
		METH_VARARGS, "Create a cuvid object."
	}, {
		"cuvid_start", (PyCFunction)CuvidStart,
		METH_VARARGS, "Please make sure current status is STANDBY."
	}, {
		"cuvid_stop", (PyCFunction)CuvidStop,
		METH_O, "Stop streaming and clear fail status."
	}, {
		"cuvid_set_jpeg_quality", (PyCFunction)CuvidSetJpegQuality,
		METH_VARARGS, "[JPEG quality] 1~100"
	}, {
		"cuvid_get_current_status", (PyCFunction)CuvidGetCurrentStatus,
		METH_O, "[Status] 0: STANDBY, 1: WORKING, -1: FAILED"
	}, {
		"cuvid_get_frame", (PyCFunction)CuvidGetFrame,
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
