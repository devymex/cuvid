from . import cuvid

class Cuvid:
    def __init__(self, gpu_id):
        self._cuvid = cuvid.cuvid_create(gpu_id)

    def open(self, url, read_mode = 0):
        return cuvid.cuvid_open(self._cuvid, url, read_mode)

    def close(self):
        cuvid.cuvid_close(self._cuvid)
    
    def get(self, propId):
        return cuvid.cuvid_get(self._cuvid, propId)

    def errcode(self):
        return cuvid.cuvid_errcode(self._cuvid)

    def read(self):
        frame_id, time_stamp, img = cuvid.cuvid_read(self._cuvid)
        return (frame_id, time_stamp, img)
