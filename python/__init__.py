#!/usr/bin/python3

from . import cuvid
import time

class Cuvid:
	def __init__(self, gpu_id):
		self._cuvid = cuvid.cuvid_create(gpu_id)

	def open(self, url):
		return cuvid.cuvid_open(self._cuvid, url)

	def close(self):
		cuvid.cuvid_close(self._cuvid)
	
	def get(self, propId):
		return cuvid.cuvid_get(self._cuvid, propId)

	def set_jpeg_quality(self, quality):
		cuvid.cuvid_set_jpeg_quality(self._cuvid, quality)

	def status(self):
		return cuvid.cuvid_status(self._cuvid)

	def read(self, with_jpeg = True):
		if with_jpeg:
			frame_id, img, jpeg = cuvid.cuvid_read(self._cuvid, True)
			return (frame_id, img, jpeg)
		else:
			frame_id, img = cuvid.cuvid_read(self._cuvid, False)
			return (frame_id, img)
