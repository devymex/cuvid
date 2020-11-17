#!/usr/bin/python3
import sys, prand, cv2, numpy as np

assert(len(sys.argv) >= 2)
rtsp_url = sys.argv[1]
gpu_id = 0
if len(sys.argv) > 2:
	gpu_id = int(sys.argv[2])

max_size = 480
def limit_size(size):
	if size[1] >= size[0] and size[1] > max_size:
		return (int((size[0] * max_size) / size[1]), max_size)
	if size[0] >= size[1] and size[0] > max_size:
		return (max_size, int((size[1] * max_size) / size[0]))
	return size

dec = prand.Prand(rtsp_url, gpu_id)
dec.set_jpeg_quality(15)
dec.start()

while True:
	frame_id, img1, jpeg = dec.get_frame(True)
	if frame_id > 0:
		img1 = cv2.resize(img1, limit_size((img1.shape[1], img1.shape[0])))
		cv2.imshow("Downloaded from GPU", img1)

		jpeg = np.frombuffer(jpeg, dtype="uint8")
		img2 = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
		img2 = cv2.resize(img2, limit_size((img2.shape[1], img2.shape[0])))
		cv2.imshow("Decode From JPEG", img2)

		key = cv2.waitKey(1) & 0xFF
		if key == 27:
			break

dec.stop()
