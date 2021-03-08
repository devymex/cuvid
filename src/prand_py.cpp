#include <Python.h>
#include <glog/logging.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "prand.hpp"

extern "C" {

void PrandDestroy(PyObject *pCapsule) {
#ifdef NDEBUG
	LOG(INFO) << "Prand Destructed";
#endif
	delete (Prand*)PyCapsule_GetPointer(pCapsule, "Prand");
}

PyObject* PrandCreate(PyObject *self, PyObject *pArgs) {
	int nDevID = 0;
	CHECK(PyArg_ParseTuple(pArgs, "i", &nDevID));

#ifdef NDEBUG
	LOG(INFO) << "Prand Created, DevID=nDevID";
#endif

	Prand *pPrand = new Prand(nDevID);
	return PyCapsule_New((void*)pPrand, "Prand", PrandDestroy);
}

PyObject* PrandStart(PyObject *self, PyObject *pArgs) {
	PyObject *pObj;
	char *pURL = nullptr;
	CHECK(PyArg_ParseTuple(pArgs, "Os", &pObj, &pURL));
	auto pPrand = (Prand*)PyCapsule_GetPointer(pObj, "Prand");
	CHECK_NOTNULL(pPrand);

	PyObject *pyResult = Py_False;
	auto [nRet, frameSize] = pPrand->Start(pURL);
	if (nRet) {
		pyResult = Py_True;
	}
	auto pyFrameSize = PyTuple_Pack(2, PyLong_FromLong(frameSize.width),
						PyLong_FromLong(frameSize.height));
	auto pyRet = PyTuple_Pack(2, pyResult, pyFrameSize);
	Py_XDECREF(pyFrameSize);
	return pyRet;
}

PyObject* PrandStop(PyObject *self, PyObject *pCapsule) {
	auto pPrand = (Prand*)PyCapsule_GetPointer(pCapsule, "Prand");
	CHECK_NOTNULL(pPrand);
	pPrand->Stop();
	Py_RETURN_NONE;
}

PyObject* PrandGetCurrentStatus(PyObject *self, PyObject *pCapsule) {
	auto pPrand = (Prand*)PyCapsule_GetPointer(pCapsule, "Prand");
	CHECK_NOTNULL(pPrand);
	auto status = pPrand->GetCurrentStatus();
	long nStatus = (status == Prand::STATUS::FAILED) ? -1 : (long)status;
	return PyLong_FromLong(nStatus);
}

PyObject* PrandSetJpegQuality(PyObject *self, PyObject *pArgs) {
	PyObject *pObj;
	int nQuality = -1;
	CHECK(PyArg_ParseTuple(pArgs, "Oi", &pObj, &nQuality));

	auto pPrand = (Prand*)PyCapsule_GetPointer(pObj, "Prand");
	CHECK_NOTNULL(pPrand);
	pPrand->SetJpegQuality(nQuality);
	Py_RETURN_NONE;
}

PyObject* NDArrayFromData(const std::vector<long> &shape, uint8_t *pData) {
	PyObject *pTmp = PyArray_SimpleNewFromData(shape.size(),
			shape.data(), NPY_UBYTE, pData);
	PyObject *pRet = PyArray_NewCopy((PyArrayObject*)pTmp, NPY_ANYORDER);
	Py_XDECREF(pTmp);
	return pRet;
}

PyObject* PrandGetFrame(PyObject *self, PyObject *pArgs) {
	PyObject *pObj;
	int nWithJpeg = 0;
	CHECK(PyArg_ParseTuple(pArgs, "O|i", &pObj, &nWithJpeg));

	auto pPrand = (Prand*)PyCapsule_GetPointer(pObj, "Prand");
	CHECK_NOTNULL(pPrand);
	cv::cuda::GpuMat gpuImg;
	std::string strJpeg;
	int64_t nFrameCnt = pPrand->GetFrame(gpuImg,
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

static PyMethodDef prand_methods[] = { 
	{
		"prand_create", (PyCFunction)PrandCreate,
		METH_VARARGS, "Create a prand object."
	}, {
		"prand_start", (PyCFunction)PrandStart,
		METH_VARARGS, "Please make sure current status is STANDBY."
	}, {
		"prand_stop", (PyCFunction)PrandStop,
		METH_O, "Stop streaming and clear fail status."
	}, {
		"prand_set_jpeg_quality", (PyCFunction)PrandSetJpegQuality,
		METH_VARARGS, "[JPEG quality] 1~100"
	}, {
		"prand_get_current_status", (PyCFunction)PrandGetCurrentStatus,
		METH_O, "[Status] 1: STANDBY, 2: WORKING, -1: FAILED"
	}, {
		"prand_get_frame", (PyCFunction)PrandGetFrame,
		METH_VARARGS, "[Return code] 0: Empty -1: Failed, >0: Successed"
	},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef prand_definition = { 
	PyModuleDef_HEAD_INIT,
	"prand",
	"Python RTSP AV Nvidia Decoder",
	-1,
	prand_methods
};

PyMODINIT_FUNC PyInit_prand(void) {
	FLAGS_alsologtostderr = 1;
	google::InitGoogleLogging("");

	Py_Initialize();
	import_array();
	return PyModule_Create(&prand_definition);
}

}
