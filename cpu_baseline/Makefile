CXX = cl
CXXFLAGS = /EHsc /O2 /std:c++17 /I C:\opencv\build\install\include

LDFLAGS = C:\opencv\build\install\x64\vc17\lib\opencv_world4130.lib

TARGET = cpub.exe
SRC = cpub.cpp

all:
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) /Fe$(TARGET)

clean:
	del $(TARGET) *.obj