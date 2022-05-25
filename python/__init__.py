from . import cuvid

class Cuvid:
    def __init__(self, gpu_id):
        self._cuvid = cuvid.cuvid_create(gpu_id)
        self.gpu_id = gpu_id

    def open(self, url, read_mode = 0):
        return cuvid.cuvid_open(self._cuvid, url, read_mode)

    def close(self):
        cuvid.cuvid_close(self._cuvid)

    def get(self, propId):
        return cuvid.cuvid_get(self._cuvid, propId)

    def errcode(self):
        return cuvid.cuvid_errcode(self._cuvid)

    def frame_shape(self):
        return (int(self.get(4)), int(self.get(3)), int(self.get(6)))

    def read_as_numpy(self):
        frame_id, time_stamp, img = cuvid.cuvid_read_as_numpy(self._cuvid)
        return (frame_id, time_stamp, img)

    def read_to_tensor(self, tensor = None):
        import torch
        if not hasattr(self, 'device'):
            self.device = torch.device(f'cuda:{self.gpu_id}')
        if tensor is None:
            tensor = torch.empty(self.frame_shape(), dtype=torch.uint8, device=self.device)
        else:
            if tensor.device != self.device:
                tensor = tensor.to(self.device)
            if tensor.dtype != torch.uint8:
                tensor = tensor.to(torch.uint8)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensor_size = tensor.element_size() * tensor.nelement()
        frame_id, time_stamp = cuvid.cuvid_read_to_buffer(self._cuvid,
                tensor.data_ptr(), tensor_size)
        return (frame_id, time_stamp, tensor)