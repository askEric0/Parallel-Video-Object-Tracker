CXX      = g++-11
# OpenCV from the fslpython environment (same as your tracker project)
OPENCV_INC = /usr/local/depot/fsl/fslpython/envs/fslpython/include/opencv4
OPENCV_LIB = /usr/local/depot/fsl/fslpython/envs/fslpython/lib

CXXFLAGS = -O3 -std=c++17 -Wall -I$(OPENCV_INC)
# Need tracking/video modules because cpub.cpp uses cv::TrackerCSRT
LDFLAGS  = -L$(OPENCV_LIB) \
           -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio \
           -lopencv_video -lopencv_tracking

TARGET   = cpub
SRC      = cpub.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET) *.o