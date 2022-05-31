#!/usr/bin/python3
import sys, cuvid

gpu_id = 0
rtsp_url = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == "--torch":
    import torch
    use_torch = True
    buf = None
else:
    import numpy as np
    use_torch = False

dec = cuvid.Cuvid(gpu_id)
ret = dec.open(rtsp_url)
print(dec.frame_shape())

while True:
    if use_torch:
        frame_id, timestamp, buf = dec.read_to_tensor(buf)
    else:
        frame_id, timestamp, buf = dec.read_as_numpy()

    if frame_id < 0:
        break
    print(f'[{frame_id}] time: {timestamp}, shape: {buf.shape}')

dec.close()
