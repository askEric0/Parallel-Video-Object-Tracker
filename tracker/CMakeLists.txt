
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-11)
project(VideoNccTracker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    # Try alternative paths
    set(OpenCV_DIR "/usr/local/depot/fsl/fslpython/envs/fslpython/lib/cmake/opencv4")
    find_package(OpenCV REQUIRED)
endif()

add_executable(tracker
    src/main.cpp
    src/baseline_kernel.cu
)

target_include_directories(tracker PRIVATE include)
target_link_libraries(tracker PRIVATE ${OpenCV_LIBS})

set(FSLPYTHON_LIBDIR "/usr/local/depot/fsl/fslpython/envs/fslpython/lib")
target_link_libraries(tracker PRIVATE
    ${FSLPYTHON_LIBDIR}/libjpeg.so.9
    ${FSLPYTHON_LIBDIR}/libjasper.so.1
    ${FSLPYTHON_LIBDIR}/libQt5Test.so.5
)

get_filename_component(GCC_LIBDIR "/usr/lib/gcc/x86_64-linux-gnu/11" ABSOLUTE)
target_link_directories(tracker PRIVATE ${GCC_LIBDIR})

set_target_properties(tracker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # Override CMake's default forwarding flag and add our custom flags
    CUDA_NVCC_FLAGS "--no-forward-unknown-to-host-compiler;-D_Float32=float;-D_Float64=double;-D_Float32x=float;-D_Float64x=double;-D_Float128=long double"
)

set(OPENCV_LIBDIR "/usr/local/depot/fsl/fslpython/envs/fslpython/lib")
add_custom_command(TARGET tracker POST_BUILD
    COMMAND chrpath -r "${GCC_LIBDIR}:${OPENCV_LIBDIR}" $<TARGET_FILE:tracker> 2>/dev/null || true
    COMMENT "Setting library search path: system libstdc++ first, then OpenCV"
)
