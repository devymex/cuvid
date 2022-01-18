#!/usr/bin/python3
import sys, cuvid

gpu_id = 0
rtsp_url = sys.argv[1]

dec = cuvid.Cuvid(gpu_id)
ret = dec.open(rtsp_url)

while True:
    frame_id, timestamp, img = dec.read()
    if frame_id < 0:
        break
    print(f'[{frame_id}] time: {timestamp}, size: {img.shape[1]}x{img.shape[0]}')

dec.close()
