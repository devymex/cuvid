## 1. Brief

Python RTSP A/V Nvidia Decoder (PRAND)

Wang Yumeng (devymex@gmai.com)

A Python library that Decoding RTSP Stream to RGB frames with GPU.

## 2. Dependencies

### 2.1 Building System

Install Basic Components

```bash
sudo apt update
sudo apt install build-essential pkg-config gdb libgoogle-glog-dev \
	libgflags-dev libssl-dev libopenblas-dev libeigen3-dev libtbb-dev \
	locate tmux git wget unzip htop net-tools autossh ffmpeg vim \
	software-properties-common
```

Install g++-7

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y gcc-7 g++-7
sudo update-alternatives \
	--install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
	--slave /usr/bin/g++ g++ /usr/bin/g++-7 \
	--slave /usr/bin/gcov gcov /usr/bin/gcov-7 \
	--slave /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-7 \
	--slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-7 \
	--slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-7 \
	--slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-7
```

Update CMake to New Version

```bash
sudo apt remove cmake
wget https://github.com/Kitware/CMake/releases/download/v3.18.4/cmake-3.18.4.tar.gz
tar -xvf cmake-3.18.4.tar.gz
cd cmake-3.18.4
./bootstrap --parallel=8
make -j8
sudo make install -j8
```

### 2.2. CUDA

确保 cuda 为 10.2 版本，且安装在 `/usr/local/cuda` 。

### 2.3. FFmpeg

```bash
sudo apt install libavcodec-dev libavformat-dev libswscale-dev
sudo apt install ffmpeg
```

### 2.4. Python3

```bash
sudo apt install python3 python3-dev python3-pip python3-numpy
```

### 2.5. OpenCV

```
sudo apt install libgtk2.0-dev
sudo apt install libtbb2 libtbb-dev libjpeg-dev libpng-dev

wget https://github.com/opencv/opencv/archive/4.3.0.zip
wget https://github.com/opencv/opencv_contrib/archive/4.3.0.zip
unzip opencv-4.3.0.zip
unzip opencv_contrib-4.3.0.zip

cd opencv-4.3.0
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON \
	-DCMAKE_INSTALL_PREFIX=/usr/local -DENABLE_CXX11=ON -DWITH_1394=OFF \
	-DCMAKE_SKIP_BUILD_RPATH=OFF -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF \
	-DWITH_IPP=ON -DWITH_TBB=ON -DWITH_OPENMP=ON -DWITH_PTHREADS_PF=ON \
	-DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DOPENCV_GENERATE_PKGCONFIG=ON \
	-DBUILD_opencv_python3=ON \
	-DPYTHON3_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')), end='')") \
	-DPYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include(), end='')") \
	-DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(), end='')") \
	-DWITH_CUDA=ON -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON -DWITH_CUBLAS=ON \
	-DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.3.0/modules
make all -j8
sudo make install -j8
```

## 3. Build

```bash
sudo updatedb
cd prand
mkdir build
cd build
cmake ..
make all -j8
make install
```

## 4. Run Test

```bash
cd prand
./build/prand_test
./test/test.py
```

按 ESC 退出测试程序。
