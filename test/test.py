#!/usr/bin/python3

import prand, cv2

dec = prand.Prand('rtsp://10.201.105.94/user=admin&password=&channel=1&stream=0.sdp', 1)
dec.start()
while True:
	frame_id, img = dec.get_frame()
	if frame_id > 0:
		img = cv2.resize(img, (960, 540))
		cv2.imshow('img', img)
		key = cv2.waitKey(0) & 0xFF
		if key == 27:
			break
dec.stop()