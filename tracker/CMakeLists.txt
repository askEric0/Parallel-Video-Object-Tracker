
cmake_minimum_required(VERSION 3.18)

# Force nvcc to use an older GCC that is compatible with CUDA 11.7.
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-11)

project(VideoNccTracker LANGUAGES CXX CUDA)

# Work around newer glibc headers that use _Float* types unsupported by
# this CUDA front-end by mapping them to regular floating-point types.
# Also disable CMake's use of -forward-unknown-to-host-compiler which
# triggers an nvcc bug with dependency flags (-MD/-MT) on some versions.
# We'll set these flags on the target itself to override CMake defaults

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find OpenCV - check common locations if not found automatically
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    # Try alternative paths
    set(OpenCV_DIR "/usr/local/depot/fsl/fslpython/envs/fslpython/lib/cmake/opencv4")
    find_package(OpenCV REQUIRED)
endif()

add_executable(tracker
    src/main.cpp
    src/baseline_kernel.cu
    src/ncc_cpu.cpp
)

target_include_directories(tracker PRIVATE include)

target_link_libraries(tracker PRIVATE ${OpenCV_LIBS})

# Explicitly link image/Qt dependencies that OpenCV imgcodecs/highgui expect.
# These live inside the same fslpython environment as OpenCV and are not
# pulled in automatically by the linker on this system.
set(FSLPYTHON_LIBDIR "/usr/local/depot/fsl/fslpython/envs/fslpython/lib")
target_link_libraries(tracker PRIVATE
    ${FSLPYTHON_LIBDIR}/libjpeg.so.9
    ${FSLPYTHON_LIBDIR}/libjasper.so.1
    ${FSLPYTHON_LIBDIR}/libQt5Test.so.5
)

# Force linking against system libstdc++ and libgcc_s instead of fslpython versions
# This fixes GLIBCXX version compatibility issues
get_filename_component(GCC_LIBDIR "/usr/lib/gcc/x86_64-linux-gnu/11" ABSOLUTE)
target_link_directories(tracker PRIVATE ${GCC_LIBDIR})

set_target_properties(tracker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # Override CMake's default forwarding flag and add our custom flags
    # Using CUDA_NVCC_FLAGS property ensures these flags come after CMake's defaults
    CUDA_NVCC_FLAGS "--no-forward-unknown-to-host-compiler;-D_Float32=float;-D_Float64=double;-D_Float32x=float;-D_Float64x=double;-D_Float128=long double"
)

# Post-build: Fix RUNPATH to prioritize system libstdc++ over fslpython
# OpenCV needs fslpython lib path, but we need system libstdc++ (which has GLIBCXX_3.4.32)
# Put GCC libdir FIRST so system libstdc++ is found before fslpython's
set(OPENCV_LIBDIR "/usr/local/depot/fsl/fslpython/envs/fslpython/lib")
add_custom_command(TARGET tracker POST_BUILD
    COMMAND chrpath -r "${GCC_LIBDIR}:${OPENCV_LIBDIR}" $<TARGET_FILE:tracker> 2>/dev/null || true
    COMMENT "Setting library search path: system libstdc++ first, then OpenCV"
)
