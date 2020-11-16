#include <Python.h>
#include <glog/logging.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "prand.hpp"

extern "C" {

void PrandDestroy(PyObject *pCapsule) {
	LOG(INFO) << "Prand Destructed";
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
	int nQuality = -1;
	CHECK(PyArg_ParseTuple(pArgs, "Oi", &pObj, &nQuality));

	auto pPrand = (Prand*)PyCapsule_GetPointer(pObj, "Prand");
	CHECK_NOTNULL(pPrand);
	pPrand->SetJpegQuality(nQuality);
	Py_RETURN_NONE;
}

PyObject* PrandGetFrame(PyObject *self, PyObject *pArgs) {
	PyObject *pObj;
	int nWithJpeg = 0;
	CHECK(PyArg_ParseTuple(pArgs, "O|i", &pObj, &nWithJpeg));

	auto pPrand = (Prand*)PyCapsule_GetPointer(pObj, "Prand");
	CHECK_NOTNULL(pPrand);
	cv::cuda::GpuMat gpuImg;
	std::string strJpeg;

	int64_t nFrameCnt;
	if (nWithJpeg > 0) {
		nFrameCnt = pPrand->GetFrame(gpuImg, &strJpeg);
		cv::Mat img;
		if (!gpuImg.empty()) {
			gpuImg.download(img);
		}
		PyObject *pNpImg = Py_None;
		PyObject *pJpeg = Py_None;
		PyObject *pTmp = nullptr;
#if 0 // OUTPUT AS NUMPY
		if (!img.empty()) {
			npy_intp dimsImg[3] = { img.rows, img.cols, img.channels() };
			pTmp = PyArray_SimpleNewFromData(3, dimsImg, NPY_UBYTE, img.data);
			pNpImg = PyArray_NewCopy((PyArrayObject*)pTmp, NPY_ANYORDER);
			Py_DECREF(pTmp);

			npy_intp dimsJpeg[1] = { (int)strJpeg.size() };
			pTmp = PyArray_SimpleNewFromData(1, dimsJpeg,
					NPY_UBYTE, (void*)strJpeg.data());
			pJpeg = PyArray_NewCopy((PyArrayObject*)pTmp, NPY_ANYORDER);
			Py_DECREF(pTmp);

		} else {
			Py_INCREF(pNpImg);
			Py_INCREF(pJpeg);
		}
#else //OUTPUT AS BYTES
		if (!img.empty()) {
			npy_intp dimsImg[3] = { img.rows, img.cols, img.channels() };
			pTmp = PyArray_SimpleNewFromData(3, dimsImg, NPY_UBYTE, img.data);
			pNpImg = PyArray_NewCopy((PyArrayObject*)pTmp, NPY_ANYORDER);
			Py_DECREF(pTmp);
		} else {
			Py_INCREF(pNpImg);
		}
		if (!strJpeg.empty()) {
			pJpeg = PyBytes_FromStringAndSize(strJpeg.data(), strJpeg.size());
		} else {
			Py_INCREF(pJpeg);
		}
#endif
		PyObject *pRet = PyTuple_Pack(3, PyLong_FromLong(nFrameCnt), pNpImg, pJpeg);
		Py_XDECREF(pJpeg);
		Py_XDECREF(pNpImg);
		return pRet;
	} else {
		nFrameCnt = pPrand->GetFrame(gpuImg, &strJpeg);
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
		PyObject *pRet = PyTuple_Pack(2, PyLong_FromLong(nFrameCnt), pNpImg);
		Py_XDECREF(pNpImg);
		return pRet;
	}
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
		"prand_get_frame", PrandGetFrame, METH_VARARGS,
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
