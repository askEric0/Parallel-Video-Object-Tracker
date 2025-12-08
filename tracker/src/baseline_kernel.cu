
#include "baseline_kernel.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstring>

constexpr int MAX_TEMPL_PIXELS = 4096;
__constant__ float d_templ_const[MAX_TEMPL_PIXELS];

namespace {
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << msg << "): "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Naive NCC kernel: each thread computes one output NCC value independently.
__global__
void nccKernelNaive(const float* frame, int frameW, int frameH, const float* templ, int templW, int templH,
                    float templMean, float templStd, float* out, int outW, int outH)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= outW || oy >= outH) return;

    const int N = templW * templH;
    float sum = 0.0f;
    float sumSq = 0.0f;

    for (int dy = 0; dy < templH; ++dy) {
        int fy = oy + dy;
        int frameRow = fy * frameW;
        for (int dx = 0; dx < templW; ++dx) {
            float v = frame[frameRow + (ox + dx)];
            sum   += v;
            sumSq += v * v;
        }
    }

    float mean = sum / N;
    float var  = sumSq / N - mean * mean;
    float std  = sqrtf(fmaxf(var, 1e-6f));

    float inv_std    = 1.0f / (std + 1e-6f);
    float inv_t_std  = 1.0f / (templStd + 1e-6f);
    float cov = 0.0f;
    for (int dy = 0; dy < templH; ++dy) {
        int fy = oy + dy;
        int frameRow = fy * frameW;
        int templRow = dy * templW;
        for (int dx = 0; dx < templW; ++dx) {
            float fv = frame[frameRow + (ox + dx)];
            float tv = templ[templRow + dx];
            cov += (fv - mean) * (tv - templMean);
        }
    }

    float ncc = cov * inv_std * inv_t_std / static_cast<float>(N);
    out[oy * outW + ox] = ncc;
}

// Same computation as nccKernelNaive, but the template is first loaded into shared memory once per block to reduce global memory traffic.
__global__
void nccKernelShared(const float* frame, int frameW, int frameH, const float* templ, int templW, int templH,
                     float templMean, float templStd, float* out, int outW, int outH)
{
    extern __shared__ float shTempl[];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    int templSize = templW * templH;
    for (int i = tid; i < templSize; i += numThreads) {
        shTempl[i] = templ[i];
    }
    __syncthreads();

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= outW || oy >= outH) return;

    const int N = templSize;

    float sum = 0.0f;
    float sumSq = 0.0f;

    for (int dy = 0; dy < templH; ++dy) {
        int fy = oy + dy;
        int frameRow = fy * frameW;
        int templRow = dy * templW;
        for (int dx = 0; dx < templW; ++dx) {
            float v = frame[frameRow + (ox + dx)];
            sum   += v;
            sumSq += v * v;
        }
    }

    float mean = sum / N;
    float var  = sumSq / N - mean * mean;
    float std  = sqrtf(fmaxf(var, 1e-6f));

    float inv_std   = 1.0f / (std + 1e-6f);
    float inv_t_std = 1.0f / (templStd + 1e-6f);

    float cov = 0.0f;
    for (int dy = 0; dy < templH; ++dy) {
        int fy = oy + dy;
        int frameRow = fy * frameW;
        int templRow = dy * templW;
        for (int dx = 0; dx < templW; ++dx) {
            float fv = frame[frameRow + (ox + dx)];
            float tv = shTempl[templRow + dx];
            cov += (fv - mean) * (tv - templMean);
        }
    }

    float ncc = cov * inv_std * inv_t_std / static_cast<float>(N);
    out[oy * outW + ox] = ncc;
}

// Batched naive kernel: blockIdx.z selects which frame in the batch.
__global__
void nccKernelNaiveBatched(const float* frames, int frameW, int frameH, const float* templ, int templW, int templH, float templMean, float templStd,
                           float* out, int outW, int outH, int numFrames)
{
    int f = blockIdx.z;
    if (f >= numFrames) return;

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= outW || oy >= outH) return;

    const int N = templW * templH;

    const float* frame = frames + static_cast<size_t>(f) * frameW * frameH;
    float* outFrame = out + static_cast<size_t>(f) * outW * outH;

    float sum = 0.0f;
    float sumSq = 0.0f;

    for (int dy = 0; dy < templH; ++dy) {
        int fy = oy + dy;
        int frameRow = fy * frameW;
        for (int dx = 0; dx < templW; ++dx) {
            float v = frame[frameRow + (ox + dx)];
            sum   += v;
            sumSq += v * v;
        }
    }

    float mean = sum / N;
    float var  = sumSq / N - mean * mean;
    float std  = sqrtf(fmaxf(var, 1e-6f));

    float inv_std   = 1.0f / (std + 1e-6f);
    float inv_t_std = 1.0f / (templStd + 1e-6f);

    float cov = 0.0f;
    for (int dy = 0; dy < templH; ++dy) {
        int fy = oy + dy;
        int frameRow = fy * frameW;
        int templRow = dy * templW;
        for (int dx = 0; dx < templW; ++dx) {
            float fv = frame[frameRow + (ox + dx)];
            float tv = templ[templRow + dx];
            cov += (fv - mean) * (tv - templMean);
        }
    }

    float ncc = cov * inv_std * inv_t_std / static_cast<float>(N);
    outFrame[oy * outW + ox] = ncc;
}

// NCC kernel using constant memory for the template.
__global__
void nccKernelConst(const float* frame, int frameW, int frameH, int templW, int templH,
                    float templMean, float templStd, float* out, int outW, int outH)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

    const int N = templW * templH;

    float sum   = 0.0f;
    float sumSq = 0.0f;

    for (int dy = 0; dy < templH; ++dy) {
        int fy        = oy + dy;
        int frameRow  = fy * frameW;
        int frameBase = frameRow + ox;
        for (int dx = 0; dx < templW; ++dx) {
            float v = frame[frameBase + dx];
            sum   += v;
            sumSq += v * v;
        }
    }

    float mean = sum / N;
    float var  = sumSq / N - mean * mean;
    float std  = sqrtf(fmaxf(var, 1e-6f));

    float inv_std   = 1.0f / (std + 1e-6f);
    float inv_t_std = 1.0f / (templStd + 1e-6f);

    float cov = 0.0f;
    for (int dy = 0; dy < templH; ++dy) {
        int fy        = oy + dy;
        int frameRow  = fy * frameW;
        int frameBase = frameRow + ox;
        int templRow  = dy * templW;
        for (int dx = 0; dx < templW; ++dx) {
            float fv = frame[frameBase + dx];
            float tv = d_templ_const[templRow + dx];
            cov += (fv - mean) * (tv - templMean);
        }
    }

    float ncc = cov * inv_std * inv_t_std / static_cast<float>(N);
    out[oy * outW + ox] = ncc;
}

// NCC kernel using constant memory for the template and shared memory for frame tiles.
__global__
void nccKernelConstTiled(const float* frame, int frameW, int frameH, int templW, int templH,
                         float templMean, float templStd, float* out, int outW, int outH)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

    const int N = templW * templH;

    int tileOx = blockIdx.x * blockDim.x;
    int tileOy = blockIdx.y * blockDim.y;

    int tileW = blockDim.x + templW - 1;
    int tileH = blockDim.y + templH - 1;

    extern __shared__ float shFrame[];

    int tid       = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    int tileSize   = tileW * tileH;

    for (int i = tid; i < tileSize; i += numThreads) {
        int ty = i / tileW;
        int tx = i % tileW;
        int fy = tileOy + ty;
        int fx = tileOx + tx;

        if (fy >= 0 && fy < frameH && fx >= 0 && fx < frameW) {
            shFrame[i] = frame[fy * frameW + fx];
        } else {
            shFrame[i] = 0.0f;
        }
    }
    __syncthreads();

    float sum   = 0.0f;
    float sumSq = 0.0f;

    int localOx = threadIdx.x;
    int localOy = threadIdx.y;

    for (int dy = 0; dy < templH; ++dy) {
        int ly = localOy + dy;
        int tileRow = ly * tileW;
        int lxBase = localOx;
        for (int dx = 0; dx < templW; ++dx) {
            float v = shFrame[tileRow + (lxBase + dx)];
            sum   += v;
            sumSq += v * v;
        }
    }

    float mean = sum / N;
    float var  = sumSq / N - mean * mean;
    float std  = sqrtf(fmaxf(var, 1e-6f));

    float inv_std   = 1.0f / (std + 1e-6f);
    float inv_t_std = 1.0f / (templStd + 1e-6f);

    float cov = 0.0f;
    for (int dy = 0; dy < templH; ++dy) {
        int ly = localOy + dy;
        int tileRow = ly * tileW;
        int lxBase = localOx;
        int templRow = dy * templW;
        for (int dx = 0; dx < templW; ++dx) {
            float fv = shFrame[tileRow + (lxBase + dx)];
            float tv = d_templ_const[templRow + dx];
            cov += (fv - mean) * (tv - templMean);
        }
    }

    float ncc = cov * inv_std * inv_t_std / static_cast<float>(N);
    out[oy * outW + ox] = ncc;
}


} // anonymous namespace

namespace baseline {

void ncc_match_naive_cuda(const cv::Mat& frame_gray_f32,
                          const cv::Mat& templ_gray_f32,
                          cv::Mat& ncc_map)
{
    CV_Assert(frame_gray_f32.type() == CV_32FC1);
    CV_Assert(templ_gray_f32.type() == CV_32FC1);

    int frameW = frame_gray_f32.cols;
    int frameH = frame_gray_f32.rows;
    int templW = templ_gray_f32.cols;
    int templH = templ_gray_f32.rows;

    int outW = frameW - templW + 1;
    int outH = frameH - templH + 1;
    CV_Assert(outW > 0 && outH > 0);

    ncc_map.create(outH, outW, CV_32FC1);

    cv::Scalar mean, stddev;
    cv::meanStdDev(templ_gray_f32, mean, stddev);
    float templMean = static_cast<float>(mean[0]);
    float templStd  = static_cast<float>(stddev[0] + 1e-6f);

    size_t frameSize = static_cast<size_t>(frameW) * frameH * sizeof(float);
    size_t templSize = static_cast<size_t>(templW) * templH * sizeof(float);
    size_t outSize   = static_cast<size_t>(outW) * outH * sizeof(float);

    float *d_frame = nullptr, *d_templ = nullptr, *d_out = nullptr;

    checkCuda(cudaMalloc(&d_frame, frameSize), "malloc frame");
    checkCuda(cudaMalloc(&d_templ, templSize), "malloc templ");
    checkCuda(cudaMalloc(&d_out,   outSize),   "malloc out");

    checkCuda(cudaMemcpy(d_frame, frame_gray_f32.ptr<float>(), frameSize, cudaMemcpyHostToDevice), "memcpy frame");
    checkCuda(cudaMemcpy(d_templ, templ_gray_f32.ptr<float>(), templSize, cudaMemcpyHostToDevice), "memcpy templ");

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);

    nccKernelNaive<<<grid, block>>>(d_frame, frameW, frameH, d_templ, templW, templH, templMean, templStd, d_out, outW, outH);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel sync");

    checkCuda(cudaMemcpy(ncc_map.ptr<float>(), d_out, outSize, cudaMemcpyDeviceToHost), "memcpy out");

    cudaFree(d_frame);
    cudaFree(d_templ);
    cudaFree(d_out);
}

void ncc_match_shared_cuda(const cv::Mat& frame_gray_f32, const cv::Mat& templ_gray_f32, cv::Mat& ncc_map)
{
    CV_Assert(frame_gray_f32.type() == CV_32FC1);
    CV_Assert(templ_gray_f32.type() == CV_32FC1);

    int frameW = frame_gray_f32.cols;
    int frameH = frame_gray_f32.rows;
    int templW = templ_gray_f32.cols;
    int templH = templ_gray_f32.rows;

    int outW = frameW - templW + 1;
    int outH = frameH - templH + 1;
    CV_Assert(outW > 0 && outH > 0);

    ncc_map.create(outH, outW, CV_32FC1);

    cv::Scalar mean, stddev;
    cv::meanStdDev(templ_gray_f32, mean, stddev);
    float templMean = static_cast<float>(mean[0]);
    float templStd  = static_cast<float>(stddev[0] + 1e-6f);

    size_t frameSize = static_cast<size_t>(frameW) * frameH * sizeof(float);
    size_t templSize = static_cast<size_t>(templW) * templH * sizeof(float);
    size_t outSize   = static_cast<size_t>(outW) * outH * sizeof(float);

    float *d_frame = nullptr, *d_templ = nullptr, *d_out = nullptr;

    checkCuda(cudaMalloc(&d_frame, frameSize), "malloc frame");
    checkCuda(cudaMalloc(&d_templ, templSize), "malloc templ");
    checkCuda(cudaMalloc(&d_out,   outSize),   "malloc out");

    checkCuda(cudaMemcpy(d_frame, frame_gray_f32.ptr<float>(), frameSize, cudaMemcpyHostToDevice), "memcpy frame");
    checkCuda(cudaMemcpy(d_templ, templ_gray_f32.ptr<float>(), templSize, cudaMemcpyHostToDevice), "memcpy templ");

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);

    size_t shmemBytes = static_cast<size_t>(templW) * templH * sizeof(float);
    nccKernelShared<<<grid, block, shmemBytes>>>(d_frame, frameW, frameH, d_templ, templW, templH, templMean, templStd, d_out, outW, outH);
    checkCuda(cudaGetLastError(), "kernel launch (shared)");
    checkCuda(cudaDeviceSynchronize(), "kernel sync (shared)");

    checkCuda(cudaMemcpy(ncc_map.ptr<float>(), d_out, outSize, cudaMemcpyDeviceToHost), "memcpy out (shared)");

    cudaFree(d_frame);
    cudaFree(d_templ);
    cudaFree(d_out);
}

void ncc_match_naive_cuda_batched(const std::vector<cv::Mat>& frames_gray_f32, const cv::Mat& templ_gray_f32, std::vector<cv::Mat>& ncc_maps)
{
    CV_Assert(!frames_gray_f32.empty());
    CV_Assert(templ_gray_f32.type() == CV_32FC1);

    int numFrames = static_cast<int>(frames_gray_f32.size());
    int frameW = frames_gray_f32[0].cols;
    int frameH = frames_gray_f32[0].rows;

    for (const auto& f : frames_gray_f32) {
        CV_Assert(f.type() == CV_32FC1);
        CV_Assert(f.cols == frameW && f.rows == frameH);
    }

    int templW = templ_gray_f32.cols;
    int templH = templ_gray_f32.rows;

    int outW = frameW - templW + 1;
    int outH = frameH - templH + 1;
    CV_Assert(outW > 0 && outH > 0);

    ncc_maps.resize(numFrames);
    for (int i = 0; i < numFrames; ++i) {
        ncc_maps[i].create(outH, outW, CV_32FC1);
    }

    cv::Scalar mean, stddev;
    cv::meanStdDev(templ_gray_f32, mean, stddev);
    float templMean = static_cast<float>(mean[0]);
    float templStd  = static_cast<float>(stddev[0] + 1e-6f);

    size_t frameSizeSingle = static_cast<size_t>(frameW) * frameH * sizeof(float);
    size_t frameSizeAll    = frameSizeSingle * numFrames;
    size_t templSize       = static_cast<size_t>(templW) * templH * sizeof(float);
    size_t outSizeSingle   = static_cast<size_t>(outW) * outH * sizeof(float);
    size_t outSizeAll      = outSizeSingle * numFrames;

    float *d_frames = nullptr, *d_templ = nullptr, *d_out = nullptr;
    checkCuda(cudaMalloc(&d_frames, frameSizeAll), "malloc frames (batched)");
    checkCuda(cudaMalloc(&d_templ,  templSize),    "malloc templ (batched)");
    checkCuda(cudaMalloc(&d_out,    outSizeAll),   "malloc out (batched)");

    std::vector<float> h_frames(static_cast<size_t>(frameW) * frameH * numFrames);
    for (int f = 0; f < numFrames; ++f) {
        const float* src = frames_gray_f32[f].ptr<float>();
        float* dst = h_frames.data() + static_cast<size_t>(f) * frameW * frameH;
        std::memcpy(dst, src, frameSizeSingle);
    }

    checkCuda(cudaMemcpy(d_frames, h_frames.data(), frameSizeAll, cudaMemcpyHostToDevice), "memcpy frames (batched)");
    checkCuda(cudaMemcpy(d_templ, templ_gray_f32.ptr<float>(), templSize, cudaMemcpyHostToDevice), "memcpy templ (batched)");

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y, numFrames);

    nccKernelNaiveBatched<<<grid, block>>>(d_frames, frameW, frameH, d_templ, templW, templH, templMean, templStd, d_out, outW, outH, numFrames);
    checkCuda(cudaGetLastError(), "kernel launch (batched)");
    checkCuda(cudaDeviceSynchronize(), "kernel sync (batched)");

    // Copy results back
    std::vector<float> h_out(static_cast<size_t>(outW) * outH * numFrames);
    checkCuda(cudaMemcpy(h_out.data(), d_out, outSizeAll,
                         cudaMemcpyDeviceToHost),
              "memcpy out (batched)");

    for (int f = 0; f < numFrames; ++f) {
        float* src = h_out.data() + static_cast<size_t>(f) * outW * outH;
        std::memcpy(ncc_maps[f].ptr<float>(), src, outSizeSingle);
    }

    cudaFree(d_frames);
    cudaFree(d_templ);
    cudaFree(d_out);
}

void ncc_match_const(const cv::Mat& frame_gray_f32, const cv::Mat& templ_gray_f32,  cv::Mat& ncc_map)
{
    CV_Assert(frame_gray_f32.type() == CV_32FC1);
    CV_Assert(templ_gray_f32.type() == CV_32FC1);

    int frameW = frame_gray_f32.cols;
    int frameH = frame_gray_f32.rows;
    int templW = templ_gray_f32.cols;
    int templH = templ_gray_f32.rows;

    int outW = frameW - templW + 1;
    int outH = frameH - templH + 1;
    CV_Assert(outW > 0 && outH > 0);

    int templPixels = templW * templH;
    CV_Assert(templPixels <= MAX_TEMPL_PIXELS);  // ensure it fits in const mem

    ncc_map.create(outH, outW, CV_32FC1);

    cv::Scalar mean, stddev;
    cv::meanStdDev(templ_gray_f32, mean, stddev);
    float templMean = static_cast<float>(mean[0]);
    float templStd  = static_cast<float>(stddev[0] + 1e-6f);

    size_t frameSize = static_cast<size_t>(frameW) * frameH * sizeof(float);
    size_t outSize   = static_cast<size_t>(outW) * outH * sizeof(float);

    float *d_frame = nullptr, *d_out = nullptr;

    checkCuda(cudaMalloc(&d_frame, frameSize), "malloc frame");
    checkCuda(cudaMalloc(&d_out,   outSize),   "malloc out");

    checkCuda(cudaMemcpy(d_frame, frame_gray_f32.ptr<float>(), frameSize, cudaMemcpyHostToDevice), "memcpy frame");


    checkCuda(cudaMemcpyToSymbol(d_templ_const, templ_gray_f32.ptr<float>(),  templPixels * sizeof(float), 0, cudaMemcpyHostToDevice), "memcpyToSymbol templ");

    dim3 block(32, 8); 
    dim3 grid((outW + block.x - 1) / block.x,
              (outH + block.y - 1) / block.y);

    nccKernelConst<<<grid, block>>>( d_frame, frameW, frameH, templW, templH, templMean, templStd, d_out, outW, outH
    );
    checkCuda(cudaGetLastError(), "kernel launch (const)");
    checkCuda(cudaDeviceSynchronize(), "kernel sync (const)");

    checkCuda(cudaMemcpy(ncc_map.ptr<float>(), d_out, outSize,
                         cudaMemcpyDeviceToHost), "memcpy out (const)");

    cudaFree(d_frame);
    cudaFree(d_out);
}

void ncc_match_const_tiled(const cv::Mat& frame_gray_f32, const cv::Mat& templ_gray_f32,  cv::Mat& ncc_map)
{
    CV_Assert(frame_gray_f32.type() == CV_32FC1);
    CV_Assert(templ_gray_f32.type() == CV_32FC1);

    int frameW = frame_gray_f32.cols;
    int frameH = frame_gray_f32.rows;
    int templW = templ_gray_f32.cols;
    int templH = templ_gray_f32.rows;

    int outW = frameW - templW + 1;
    int outH = frameH - templH + 1;
    CV_Assert(outW > 0 && outH > 0);

    int templPixels = templW * templH;
    CV_Assert(templPixels <= MAX_TEMPL_PIXELS); 

    ncc_map.create(outH, outW, CV_32FC1);

    cv::Scalar mean, stddev;
    cv::meanStdDev(templ_gray_f32, mean, stddev);
    float templMean = static_cast<float>(mean[0]);
    float templStd  = static_cast<float>(stddev[0] + 1e-6f);

    size_t frameSize = static_cast<size_t>(frameW) * frameH * sizeof(float);
    size_t outSize   = static_cast<size_t>(outW) * outH * sizeof(float);

    float *d_frame = nullptr, *d_out = nullptr;

    checkCuda(cudaMalloc(&d_frame, frameSize), "malloc frame (tiled)");
    checkCuda(cudaMalloc(&d_out,   outSize),   "malloc out (tiled)");

    checkCuda(cudaMemcpy(d_frame, frame_gray_f32.ptr<float>(),
                         frameSize, cudaMemcpyHostToDevice), "memcpy frame (tiled)");

    checkCuda(cudaMemcpyToSymbol(d_templ_const, templ_gray_f32.ptr<float>(),  templPixels * sizeof(float), 0, cudaMemcpyHostToDevice), "memcpyToSymbol templ (tiled)");

    dim3 block(32, 8); 
    dim3 grid((outW + block.x - 1) / block.x,
              (outH + block.y - 1) / block.y);

    int tileW = block.x + templW - 1;
    int tileH = block.y + templH - 1;
    size_t shmemBytes = static_cast<size_t>(tileW) * tileH * sizeof(float);

    nccKernelConstTiled<<<grid, block, shmemBytes>>>(
        d_frame, frameW, frameH,
        templW, templH,
        templMean, templStd,
        d_out, outW, outH
    );
    checkCuda(cudaGetLastError(), "kernel launch (tiled)");
    checkCuda(cudaDeviceSynchronize(), "kernel sync (tiled)");

    checkCuda(cudaMemcpy(ncc_map.ptr<float>(), d_out, outSize, cudaMemcpyDeviceToHost), "memcpy out (tiled)");

    cudaFree(d_frame);
    cudaFree(d_out);
}


} // namespace baseline



