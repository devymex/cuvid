#include <Python.h>
#include <glog/logging.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "prand.hpp"

extern "C" {

void PrandDestroy(PyObject *pCapsule) {
	LOG(INFO) << "Destructed";
	delete (Prand*)PyCapsule_GetPointer(pCapsule, "Prand");
}

PyObject* PrandCreate(PyObject *self, PyObject *pArgs) {
	char *pURL = nullptr;
	int nDevID = 0;
	CHECK(PyArg_ParseTuple(pArgs, "s|i", &pURL, &nDevID));
	LOG(INFO) << "URL: " << pURL << ", DevID: " << nDevID;

	Prand *pPrand = new Prand(pURL, nDevID);
	return PyCapsule_New((void*)pPrand, "Prand", PrandDestroy);
}

PyObject* PrandStart(PyObject *self, PyObject *pCapsule) {
	auto pPrand = (Prand*)PyCapsule_GetPointer(pCapsule, "Prand");
	CHECK_NOTNULL(pPrand);
	pPrand->Start();
	Py_RETURN_NONE;
}

PyObject* PrandStop(PyObject *self, PyObject *pCapsule) {
	auto pPrand = (Prand*)PyCapsule_GetPointer(pCapsule, "Prand");
	CHECK_NOTNULL(pPrand);
	pPrand->Stop();
	Py_RETURN_NONE;
}

PyObject* PrandSetJpegQuality(PyObject *self, PyObject *pArgs) {
	PyObject *pObj;
	int nQuality = 0;
	CHECK(PyArg_ParseTuple(pArgs, "Oi", &pObj, &nQuality));

	auto pPrand = (Prand*)PyCapsule_GetPointer(pObj, "Prand");
	CHECK_NOTNULL(pPrand);
	pPrand->SetJpegQuality(nQuality);
	Py_RETURN_NONE;
}

PyObject* PrandGetFrame(PyObject *self, PyObject *pCapsule) {
	PyObject *pObj;
	int nWithJpeg = 0;
	CHECK(PyArg_ParseTuple(pArgs, "Oi", &pObj, &nWithJpeg));

	auto pPrand = (Prand*)PyCapsule_GetPointer(pCapsule, "Prand");
	CHECK_NOTNULL(pPrand);
	cv::cuda::GpuMat gpuImg;
	std::string strJpeg;

	int64_t nFrameCnt;
	if (nWithJpeg) {
		nFrameCnt = pPrand->GetFrame(gpuImg, &strJpeg);
	} else {
		nFrameCnt = pPrand->GetFrame(gpuImg);
	}

	cv::Mat img;
	if (!gpuImg.empty()) {
		gpuImg.download(img);
	}

	PyObject *pNpImg = Py_None;
	if (!img.empty()) {
		npy_intp dimsImg[3] = { img.rows, img.cols, img.channels() };
		pNpImg = PyArray_SimpleNewFromData(3, dimsImg, NPY_UBYTE, img.data);
	} else {
		Py_INCREF(pNpImg);
	}
	if (!strJpeg.empty()) {
		PyObject *pNpJpeg = PyArray_SimpleNewFromData(1, { (int)strJpeg.size() },
				NPY_UBYTE, (void*)strJpeg.data());
		return PyTuple_Pack(3, PyLong_FromLong(nFrameCnt), pNpImg, pNpJpeg);
	}
	return PyTuple_Pack(2, PyLong_FromLong(nFrameCnt), pNpImg);
}

static PyMethodDef prand_methods[] = { 
	{
		"prand_create", PrandCreate, METH_VARARGS,
		"Create a prand object."
	},
	{
		"prand_start", PrandStart, METH_O,
		"Start decoding."
	},
	{
		"prand_stop", PrandStop, METH_O,
		"Stop decoding."
	},
	{
		"prand_set_jpeg_quality", PrandSetJpegQuality, METH_VARARGS,
		"Set encoding quality of JPEG encoder."
	},
	{
		"prand_get_frame", PrandGetFrame, METH_O,
		"Get Frame."
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
