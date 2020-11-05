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

PyObject* PrandCreate(PyObject *self, PyObject *args) {
	char *pURL = nullptr;
	int nDevID = 0;
	CHECK(PyArg_ParseTuple(args, "s|i", &pURL, &nDevID));

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

PyObject* PrandGetFrame(PyObject *self, PyObject *pCapsule) {
	auto pPrand = (Prand*)PyCapsule_GetPointer(pCapsule, "Prand");
	CHECK_NOTNULL(pPrand);
	cv::Mat img;
	int64_t nFrameCnt = pPrand->GetFrame(img);

	PyObject *pNpImg = nullptr;
	if (!img.empty()) {
		npy_intp dims[3] = {img.rows, img.cols, img.channels()};
		pNpImg = PyArray_SimpleNewFromData(3, dims, NPY_UBYTE, img.data);
	} else {
		pNpImg = Py_None;
		Py_INCREF(pNpImg);
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
		"prand_get_frame", PrandGetFrame, METH_O,
		"GetFrame."
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
