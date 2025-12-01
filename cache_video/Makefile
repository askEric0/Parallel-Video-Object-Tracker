# CXX = cl
# CXXFLAGS = /EHsc /O2 /std:c++17 /I C:\opencv\build\install\include
# LDFLAGS = C:\opencv\build\install\x64\vc17\lib\opencv_world4130.lib

# TARGET = cache_video.exe
# SRC = cache_video.cpp

# all:
# 	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) /Fe$(TARGET)

# clean:
# 	del $(TARGET) *.obj
CXX      = g++
CXXFLAGS = -O2 -std=c++17 \
           -I/usr/local/depot/fsl/fslpython/envs/fslpython/include/opencv4
LDFLAGS  = -L/usr/local/depot/fsl/fslpython/envs/fslpython/lib \
           -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio

TARGET = cache_video
SRC    = cache_video.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET) *.o