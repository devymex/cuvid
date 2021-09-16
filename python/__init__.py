#!/usr/bin/python3

from . import cuvid
import time

class Cuvid:
	def __init__(self, gpu_id):
		self._cuvid = cuvid.cuvid_create(gpu_id)

	def open(self, url, read_mode = 0):
		return cuvid.cuvid_open(self._cuvid, url, read_mode)

	def close(self):
		cuvid.cuvid_close(self._cuvid)
	
	def get(self, propId):
		return cuvid.cuvid_get(self._cuvid, propId)

	def errcode(self):
		return cuvid.cuvid_errcode(self._cuvid)

	def read(self, with_jpeg = True):
		if with_jpeg:
			frame_id, img, jpeg = cuvid.cuvid_read(self._cuvid, True)
			return (frame_id, img, jpeg)
		else:
			frame_id, img = cuvid.cuvid_read(self._cuvid, False)
			return (frame_id, img)
