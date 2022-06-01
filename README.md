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

可以直接用 `apt-get` 命令安装 CMake：

```bash
sudo apt install cmake
cmake --version
```

但如果安装的 CMake 低于 3.0.0 版本则无法运行安装，可从源代码直接编译生成 CMake。

```bash
CMAKE_VERSION=3.23.2
sudo apt remove cmake
sudo apt install libssl-dev
wget "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz"
tar -xvf cmake-$CMAKE_VERSION.tar.gz
cd cmake-$CMAKE_VERSION
./bootstrap --parallel=8
make -j8
sudo make install -j8
```

### 2.3. CUDA

确保 CUDA 安装在 `/usr/local/cuda` ，并确认本地磁盘中存在 `libnvcuvid.so` 文件。

在 Ubuntu 系统中，通常该文件位于 `/usr/lib/x86_64-linux-gnu` 目录中。有时文件会带有版本后缀，如 `libnvcuvid.so.1` 。

如果通过搜索仍未在磁盘中找到该文件，则说明你所安装的显卡驱动不完整，应重新安装显卡驱动 (必须是 460.32.03 以上版本) ，驱动包中均带有此文件。

如果仅找到 `libnvcuvid.so.1` 而没有 `libnvcuvid.so` ，那你还需要为 ``libnvcuvid.so.1` 在相同目录下建立一个名为 `libnvcuvid.so` 的软链接。

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

在 cuvid 工程目录下执行下列命令：

```bash
mkdir build
cd build
cmake .. -DNVCUVID_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu -DCUDA_ARCHITECTURES=75 -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=DEBUG
make all -j8
sudo make install
```
注意：
* 如果 `libnvcuvid.so` 文件处于默认路径 `/usr/lib/x86_64-linux-gnu` 中，则不用设定 `NVCUVID_LIBRARY_DIR` 变量，否则必须设定 `NVCUVID_LIBRARY_DIR` 为所在路径。
* 需要根据当前显卡的类型来配置 `CUDA_ARCHITECTURES` ，详见 https://zhuanlan.zhihu.com/p/438939299 ；
* 如果你在使用 Conda 环境，请确保在执行 cmake 命令前，当前 Conda 环境处于激活状态： `conda activate` 。
* 如需使用 C++ 接口，请设置 `CMAKE_INSTALL_PREFIX` 为 `/usr/local` ；

## 4. Run Test

```bash
python3 ./test/test.py test.mp4 --torch
```

## 5. C++ Project Link to Cuvid

Add following line into your `CMakeLists.txt`:

```cmake
TARGET_LINK_LIBRARIES(YOUR_PRJ_NAME nvcuvid cuda avformat avcodec avutil)
```