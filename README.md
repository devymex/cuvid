## 1. Brief

CUDA Video Decoder (CUVID)

Wang Yumeng (devymex@gmail.com)

A video decoder library that supports H264, HEVC(H265) videos from file or RTSP.

## 2. Dependencies

### 2.1 Building System

Install Basic Components

```bash
sudo apt update
sudo apt install build-essential pkg-config
sudo apt install libgoogle-glog-dev libgflags-dev libssl-dev
```

### 2.2. CMake

```bash
sudo apt install cmake
```

Or

```bash
sudo apt remove cmake
wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4.tar.gz
tar -xvf cmake-3.18.4.tar.gz
cd cmake-3.18.4
./bootstrap --parallel=8
make -j8
sudo make install -j8
```

### 2.3. CUDA

确保 cuda 为 10.2 版本，且安装在 `/usr/local/cuda` 。

### 2.4. FFmpeg

```bash
sudo apt install libavcodec-dev libavformat-dev libswscale-dev
sudo apt install ffmpeg
```

### 2.5. Numpy (Python3)

```bash
python3 -m pip install numpy
```

### 2.6. Pytorch (Optional)

See https://pytorch.org/

## 3. Build

```bash
cd cuvid
mkdir build
cd build
cmake ..
make all -j8
sudo make install
```

## 4. Run Test

```bash
cd cuvid
./build/cuvid_test
./test/test.py
```

按 ESC 退出测试程序。

## 5. Link to Cuvid

```cmake
TARGET_LINK_LIBRARIES(YOUR_PRJ_NAME nvjpeg nvcuvid cuda avformat avcodec avutil)
```