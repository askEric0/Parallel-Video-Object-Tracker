NVCC = nvcc
OPENCV_DIR = C:\opencv\build\install
CUDA_DIR   = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9

TARGET = tracker.exe
SRC    = src\baseline_kernel.cu src\main.cpp src\ncc_cpu.cpp

all:
	$(NVCC) -std=c++17 -O2 -Xcompiler "/EHsc /O2" \
	-I"include" \
	-I"$(OPENCV_DIR)\include" \
	-I"$(CUDA_DIR)\include" \
	\
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_90,code=compute_90 \
	-gencode arch=compute_120,code=sm_120 \
	\
	$(SRC) \
	"$(OPENCV_DIR)\x64\vc17\lib\opencv_core4130.lib" \
	"$(OPENCV_DIR)\x64\vc17\lib\opencv_imgproc4130.lib" \
	"$(OPENCV_DIR)\x64\vc17\lib\opencv_highgui4130.lib" \
	"$(OPENCV_DIR)\x64\vc17\lib\opencv_videoio4130.lib" \
	"$(OPENCV_DIR)\x64\vc17\lib\opencv_video4130.lib" \
	"$(OPENCV_DIR)\x64\vc17\lib\opencv_tracking4130.lib" \
	"$(OPENCV_DIR)\x64\vc17\lib\opencv_cudev4130.lib" \
	"$(OPENCV_DIR)\x64\vc17\lib\opencv_cudaarithm4130.lib" \
	"$(OPENCV_DIR)\x64\vc17\lib\opencv_cudaoptflow4130.lib" \
	"$(CUDA_DIR)\lib\x64\cudart.lib" \
	-o $(TARGET)

clean:
	del /Q $(TARGET) *.obj
