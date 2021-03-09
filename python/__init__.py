#!/usr/bin/python3

from . import prand
import time

class Prand:
	def __init__(self, gpu_id):
		self._prand = prand.prand_create(gpu_id)

	def start(self, url):
		return prand.prand_start(self._prand, url)

	def stop(self):
		prand.prand_stop(self._prand)
	
	def set_jpeg_quality(self, quality):
		prand.prand_set_jpeg_quality(self._prand, quality)

	def get_current_status(self):
		return prand.prand_get_current_status(self._prand)

	def get_frame(self, with_jpeg = True):
		if with_jpeg:
			frame_id, img, jpeg = prand.prand_get_frame(self._prand, True)
			return (frame_id, img, jpeg)
		else:
			frame_id, img = prand.prand_get_frame(self._prand, False)
			return (frame_id, img)
