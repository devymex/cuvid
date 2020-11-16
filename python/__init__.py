#!/usr/bin/python3

from . import prand
import time

class Prand:
	def __init__(self, url, gpu_id):
		self._prand = prand.prand_create(url, gpu_id)
		self._frame_id = 0

	def start(self):
		prand.prand_start(self._prand)

	def stop(self):
		prand.prand_stop(self._prand)
	
	def set_jpeg_quality(self, quality):
		prand.prand_set_jpeg_quality(self._prand, quality)

	def get_frame(self):
		frame_id, img = prand.prand_get_frame(self._prand)
		if frame_id != self._frame_id:
			self._frame_id += frame_id
			return (self._frame_id, img)
		return (0, None)
