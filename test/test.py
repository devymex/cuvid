#!/usr/bin/python3
import sys, cuvid, torch

gpu_id = 0
use_torch = False

rtsp_url = sys.argv[1]

dec = cuvid.Cuvid(gpu_id)
ret = dec.open(rtsp_url)
print(dec.frame_shape())

if use_torch:
    import torch
    buf = torch.zeros(dec.frame_shape(), dtype=torch.uint8).to(f'cuda:{gpu_id}')

while True:
    if use_torch:
       frame_id, timestamp = dec.read_to_tensor(buf)
    else:
        frame_id, timestamp, buf = dec.read_as_numpy()

    if frame_id < 0:
        break
    print(f'[{frame_id}] time: {timestamp}, shape: {buf.shape}')

dec.close()
