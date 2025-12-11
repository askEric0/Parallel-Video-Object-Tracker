CXX = cl

OPENCV_DIR = C:\opencv\build\install

CXXFLAGS = /EHsc /O2 /std:c++17 /I "$(OPENCV_DIR)\include"

LDFLAGS = ^
"$(OPENCV_DIR)\x64\vc17\lib\opencv_core4130.lib" ^
"$(OPENCV_DIR)\x64\vc17\lib\opencv_imgproc4130.lib" ^
"$(OPENCV_DIR)\x64\vc17\lib\opencv_highgui4130.lib" ^
"$(OPENCV_DIR)\x64\vc17\lib\opencv_videoio4130.lib" ^
"$(OPENCV_DIR)\x64\vc17\lib\opencv_video4130.lib" ^
"$(OPENCV_DIR)\x64\vc17\lib\opencv_tracking4130.lib" ^
"$(OPENCV_DIR)\x64\vc17\lib\opencv_cudev4130.lib" ^
"$(OPENCV_DIR)\x64\vc17\lib\opencv_cudaarithm4130.lib" ^
"$(OPENCV_DIR)\x64\vc17\lib\opencv_cudaoptflow4130.lib"

TARGET = cudab.exe
SRC    = cudab.cpp

all:
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) /Fe$(TARGET)

clean:
	del /Q $(TARGET) *.obj
