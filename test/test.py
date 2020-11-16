#!/usr/bin/python3

import prand, cv2

dec = prand.Prand('rtsp://10.201.105.94/user=admin&password=&channel=1&stream=0.sdp', 1)
dec.start()
while True:
	frame_id, img1, jpg = dec.get_frame()
	if frame_id > 0:
		img1 = cv2.resize(img1, (960, 540))
		img2 = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
		cv2.imshow("Downloaded from GPU", img1)
		cv2.imshow("Decode From JPEG", img2)
		key = cv2.waitKey(0) & 0xFF
		if key == 27:
			break
dec.stop()
