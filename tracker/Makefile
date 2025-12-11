EXECUTABLE := tracker
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
CU_FILES   := baseline_kernel.cu
CU_DEPS    :=
CC_FILES   := main.cpp
LOGS	   := logs

all: $(EXECUTABLE)


ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++-11 -m64
CXXFLAGS=-O3 -Wall -g -std=c++17
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/g++-11 --no-forward-unknown-to-host-compiler -D_Float32=float -D_Float64=double -D_Float32x=float -D_Float64x=double -D_Float128=long double
LIBS += opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_videoio cudart

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/baseline_kernel.o

INCLUDES=-Iinclude -I/usr/local/include/opencv4

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)

export: $(EXFILES)
	cp -p $(EXFILES) $(STARTER)


$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)




$(OBJDIR)/%.o: src/%.cpp
		$(CXX) $< $(CXXFLAGS) $(INCLUDES) -c -o $@

$(OBJDIR)/%.o: src/%.cu
		$(NVCC) $< $(NVCCFLAGS) $(INCLUDES) -c -o $@

