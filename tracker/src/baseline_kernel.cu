#include "baseline_kernel.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstring>

constexpr int MAX_TEMPL_PIXELS = 4096;
__constant__ float templ_const[MAX_TEMPL_PIXELS];

__global__ void nccKernelNaive(const float* frame, int frameW, int frameH, const float* templ, int templW, int templH,
                               float templMean, float templStd,  float* out, int outW, int outH)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ox >= outW || oy >= outH) return;

    const int N = templW * templH;
    float sum = 0.0f;
    float ssq = 0.0f;
    for (int dy = 0; dy < templH; dy++) {
        int idx = (oy + dy) * frameW + ox;
        for (int dx = 0; dx < templW; dx++) {
            float val = frame[idx + dx];
            sum += val;
            ssq += val * val;
        }
    }
    float frame_mean = sum / N;
    float var = ssq / N - frame_mean * frame_mean;
    float std = sqrtf(fmaxf(var, 1e-6f)); 
    float cov = 0.0f;
    for (int dy = 0; dy < templH; dy++) {
        int idx = (oy + dy) * frameW + ox;
        int templ_idx = dy * templW;
        for (int dx = 0; dx < templW; dx++) {
            float frame_val = frame[idx + dx];
            float templ_val = templ[templ_idx + dx];
            cov += (frame_val - frame_mean) * (templ_val - templMean);
        }
    }
    float ncc = cov / ((std + 1e-6f) * (templStd + 1e-6f) * (float)(N));
    out[oy * outW + ox] = ncc;
}

__global__ void nccKernelShared(const float* frame, int frameW, int frameH, const float* templ, int templW, int templH,
                     float templMean, float templStd, float* out, int outW, int outH)
{
    extern __shared__ float shared_templ[];
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    int templSize = templW * templH;
    for (int i = tid; i < templSize; i += numThreads) {
        shared_templ[i] = templ[i];
    }
    __syncthreads();

    if (ox >= outW || oy >= outH) return;
    const int N = templW * templH;
    float sum = 0.0f;
    float ssq = 0.0f;
    for (int dy = 0; dy < templH; ++dy) {
        int idx = (oy + dy) * frameW + ox;
        for (int dx = 0; dx < templW; dx++) {
            float val = frame[idx + dx];
            sum += val;
            ssq += val * val;
        }
    }
    float frame_mean = sum / N;
    float var = ssq / N - frame_mean * frame_mean;
    float std = sqrtf(fmaxf(var, 1e-6f));
    float cov = 0.0f;
    for (int dy = 0; dy < templH; dy++) {
        int idx = (oy + dy) * frameW + ox;
        int templ_idx = dy * templW;
        for (int dx = 0; dx < templW; dx++) {
            float frame_val = frame[idx + dx];
            float templ_val = shared_templ[templ_idx + dx];
            cov += (frame_val - frame_mean) * (templ_val - templMean);
        }
    }
    float ncc = cov / ((std + 1e-6f) * (templStd + 1e-6f) * (float)(N));
    out[oy * outW + ox] = ncc;
}

__global__ void nccKernelNaiveBatched(const float* frames, int frameW, int frameH, const float* templ, int templW, int templH,
                                      float templMean, float templStd, float* out, int outW, int outH, int numFrames) 
{
    int f = blockIdx.z;
    if (f >= numFrames) return;

    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;
    const int N = templW * templH;

    const float* frame = frames + (size_t)(f * frameW * frameH);
    float* outFrame = out + (size_t)(f * outW * outH);
    float sum = 0.0f;
    float ssq = 0.0f;
    for (int dy = 0; dy < templH; dy++) {
        int idx = (oy + dy) * frameW + ox;
        for (int dx = 0; dx < templW; dx++) {
            float val = frame[idx + dx];
            sum += val;
            ssq += val * val;
        }
    }
    float frame_mean = sum / N;
    float var  = ssq / N - frame_mean * frame_mean;
    float std  = sqrtf(fmaxf(var, 1e-6f));
    float cov = 0.0f;
    for (int dy = 0; dy < templH; dy++) {
        int idx = (oy + dy) * frameW + ox;
        int templ_idx = dy * templW;
        for (int dx = 0; dx < templW; dx++) {
            float frame_val = frame[idx + dx];
            float templ_val = templ[templ_idx + dx];
            cov += (frame_val - frame_mean) * (templ_val - templMean);
        }
    }
    float ncc = cov / ((std + 1e-6f) * (templStd + 1e-6f) * (float)(N));
    outFrame[oy * outW + ox] = ncc;
}

__global__ void nccKernelConst(const float* frame, int frameW, int frameH, int templW, int templH, 
                               float templMean, float templStd, float* out, int outW, int outH)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;
    const int N = templW * templH;

    float sum = 0.0f;
    float ssq = 0.0f;
    for (int dy = 0; dy < templH; dy++) {
        int idx = (oy + dy) * frameW + ox;
        for (int dx = 0; dx < templW; dx++) {
            float val = frame[idx + dx];
            sum += val;
            ssq += val * val;
        }
    }
    float frame_mean = sum / N;
    float var  = ssq / N - frame_mean * frame_mean;
    float std  = sqrtf(fmaxf(var, 1e-6f));
    float cov = 0.0f;
    for (int dy = 0; dy < templH; dy++) {
        int idx = (oy + dy) * frameW + ox;
        int templ_idx = dy * templW;
        for (int dx = 0; dx < templW; dx++) {
            float frame_val = frame[idx + dx];
            float templ_val = templ_const[templ_idx + dx];
            cov += (frame_val - frame_mean) * (templ_val - templMean);
        }
    }
    float ncc = cov / ((std + 1e-6f) * (templStd + 1e-6f) * (float)(N));
    out[oy * outW + ox] = ncc;
}
__global__ void nccKernelConstTiled(const float* frame, int frameW, int frameH, int templW, int templH,
                                    float templMean, float templStd, float* out, int outW, int outH)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

    const int N = templW * templH;
    int tileX = blockIdx.x * blockDim.x;
    int tileY = blockIdx.y * blockDim.y;
    int tileW = blockDim.x + templW - 1;
    int tileH = blockDim.y + templH - 1;
    extern __shared__ float shared_frame[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;
    int tileSize = tileW * tileH;
    for (int i = tid; i < tileSize; i += numThreads) {
        int ty = i / tileW;
        int tx = i % tileW;
        int fy = tileY + ty;
        int fx = tileX + tx;
        if (fy >= 0 && fy < frameH && fx >= 0 && fx < frameW) {
            shared_frame[i] = frame[fy * frameW + fx];
        } else {
            shared_frame[i] = 0.0f;
        }
    }
    __syncthreads();

    float sum = 0.0f;
    float ssq = 0.0f;
    int localX = threadIdx.x;
    int localY = threadIdx.y;
    for (int dy = 0; dy < templH; dy++) {
        int tile_row = (localY + dy) * tileW;
        for (int dx = 0; dx < templW; dx++) {
            float val = shared_frame[tile_row + (localX + dx)];
            sum += val;
            ssq += val * val;
        }
    }

    float frame_mean = sum / N;
    float var  = ssq / N - frame_mean * frame_mean;
    float std  = sqrtf(fmaxf(var, 1e-6f));

    float cov = 0.0f;
    for (int dy = 0; dy < templH; dy++) {
        int tile_row = (localY + dy) * tileW;
        int templ_row = dy * templW;
        for (int dx = 0; dx < templW; dx++) {
            float frame_val = shared_frame[tile_row + (localX + dx)];
            float templ_val = templ_const[templ_row + dx];
            cov += (frame_val - frame_mean) * (templ_val - templMean);
        }
    }
    float ncc = cov / ((std + 1e-6f) * (templStd + 1e-6f) * (float)(N));
    out[oy * outW + ox] = ncc;
}

void ncc_match_naive_cuda(const cv::Mat& frame_gray, const cv::Mat& templ_gray, cv::Mat& ncc_map)
{
    int frameW = frame_gray.cols;
    int frameH = frame_gray.rows;
    int templW = templ_gray.cols;
    int templH = templ_gray.rows;
    int outW = frameW - templW + 1;
    int outH = frameH - templH + 1;
    ncc_map.create(outH, outW, CV_32FC1);
    cv::Scalar mean, stddev;
    cv::meanStdDev(templ_gray, mean, stddev);
    float templMean = (float)(mean[0]);
    float templStd  = (float)(stddev[0] + 1e-6f);

    size_t frameSize = (size_t)(frameW) * frameH * sizeof(float);
    size_t templSize = (size_t)(templW) * templH * sizeof(float);
    size_t outSize = (size_t)(outW) * outH * sizeof(float);
    float *dev_frame = NULL, *dev_templ = NULL, *dev_out = NULL;

    cudaMalloc(&dev_frame, frameSize);
    cudaMalloc(&dev_templ, templSize);
    cudaMalloc(&dev_out,   outSize);
    cudaMemcpy(dev_frame, frame_gray.ptr<float>(), frameSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_templ, templ_gray.ptr<float>(), templSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
    nccKernelNaive<<<grid, block>>>(dev_frame, frameW, frameH, dev_templ, templW, templH,
                                    templMean, templStd, dev_out, outW, outH);
    cudaDeviceSynchronize();
    cudaMemcpy(ncc_map.ptr<float>(), dev_out, outSize, cudaMemcpyDeviceToHost);

    cudaFree(dev_frame);
    cudaFree(dev_templ);
    cudaFree(dev_out);
}

void ncc_match_shared_cuda(const cv::Mat& frame_gray,
                           const cv::Mat& templ_gray,
                           cv::Mat& ncc_map)
{
    int frameW = frame_gray.cols;
    int frameH = frame_gray.rows;
    int templW = templ_gray.cols;
    int templH = templ_gray.rows;
    int outW = frameW - templW + 1;
    int outH = frameH - templH + 1;

    ncc_map.create(outH, outW, CV_32FC1);
    cv::Scalar mean, stddev;
    cv::meanStdDev(templ_gray, mean, stddev);
    float templMean = (float)(mean[0]);
    float templStd  = (float)(stddev[0] + 1e-6f);

    size_t frameSize = (size_t)(frameW) * frameH * sizeof(float);
    size_t templSize = (size_t)(templW) * templH * sizeof(float);
    size_t outSize   = (size_t)(outW) * outH * sizeof(float);
    float *dev_frame = NULL, *dev_templ = NULL, *dev_out = NULL;
    cudaMalloc(&dev_frame, frameSize);
    cudaMalloc(&dev_templ, templSize);
    cudaMalloc(&dev_out,   outSize);
    cudaMemcpy(dev_frame, frame_gray.ptr<float>(), frameSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_templ, templ_gray.ptr<float>(), templSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
    size_t shmemBytes = (size_t)(templW) * templH * sizeof(float);
    nccKernelShared<<<grid, block, shmemBytes>>>(dev_frame, frameW, frameH, dev_templ, templW, templH,
                                                templMean, templStd, dev_out, outW, outH);
    cudaDeviceSynchronize();
    cudaMemcpy(ncc_map.ptr<float>(), dev_out, outSize, cudaMemcpyDeviceToHost);
    cudaFree(dev_frame);
    cudaFree(dev_templ);
    cudaFree(dev_out);
}

void ncc_match_naive_cuda_batched(const std::vector<cv::Mat>& frames_gray, const cv::Mat& templ_gray, std::vector<cv::Mat>& ncc_maps)
{
    int numFrames = (int)(frames_gray.size());
    int frameW = frames_gray[0].cols;
    int frameH = frames_gray[0].rows;
    int templW = templ_gray.cols;
    int templH = templ_gray.rows;
    int outW = frameW - templW + 1;
    int outH = frameH - templH + 1;

    ncc_maps.resize(numFrames);
    for (int i = 0; i < numFrames; ++i) {
        ncc_maps[i].create(outH, outW, CV_32FC1);
    }
    cv::Scalar mean, stddev;
    cv::meanStdDev(templ_gray, mean, stddev);
    float templMean = (float)(mean[0]);
    float templStd  = (float)(stddev[0] + 1e-6f);
    size_t frameSizeSingle = (size_t)(frameW) * frameH * sizeof(float);
    size_t frameSizeAll = frameSizeSingle * numFrames;
    size_t templSize = (size_t)(templW) * templH * sizeof(float);
    size_t outSizeSingle = (size_t)(outW) * outH * sizeof(float);
    size_t outSizeAll = outSizeSingle * numFrames;

    float *dev_frames = NULL, *dev_templ = NULL, *dev_out = NULL;
    cudaMalloc(&dev_frames, frameSizeAll);
    cudaMalloc(&dev_templ, templSize);
    cudaMalloc(&dev_out, outSizeAll);
    for (int i = 0; i < numFrames; i++) {
        cudaMemcpy(dev_frames + (size_t)(i) * frameW * frameH, frames_gray[i].ptr<float>(), 
                   frameSizeSingle, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(dev_templ, templ_gray.ptr<float>(), templSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y, numFrames);

    nccKernelNaiveBatched<<<grid, block>>>(dev_frames, frameW, frameH, dev_templ, templW, templH,
                                           templMean, templStd, dev_out, outW, outH, numFrames);
    cudaDeviceSynchronize();

    for (int i = 0; i < numFrames; i++) {
        cudaMemcpy(ncc_maps[i].ptr<float>(),  dev_out + (size_t)(i) * outW * outH,  outSizeSingle, cudaMemcpyDeviceToHost);
    }
    cudaFree(dev_frames);
    cudaFree(dev_templ);
    cudaFree(dev_out);
}

void ncc_match_const(const cv::Mat& frame_gray, const cv::Mat& templ_gray, cv::Mat& ncc_map)
{
    int frameW = frame_gray.cols;
    int frameH = frame_gray.rows;
    int templW = templ_gray.cols;
    int templH = templ_gray.rows;
    int outW = frameW - templW + 1;
    int outH = frameH - templH + 1;
    int templPixels = templW * templH;
    CV_Assert(templPixels <= MAX_TEMPL_PIXELS);
    ncc_map.create(outH, outW, CV_32FC1);
    cv::Scalar mean, stddev;
    cv::meanStdDev(templ_gray, mean, stddev);
    float templMean = (float)(mean[0]);
    float templStd = (float)(stddev[0] + 1e-6f);
    size_t frameSize = (size_t)(frameW) * frameH * sizeof(float);
    size_t outSize = (size_t)(outW) * outH * sizeof(float);
    float *dev_frame = NULL, *dev_out = NULL;
    cudaMalloc(&dev_frame, frameSize);
    cudaMalloc(&dev_out,   outSize);
    cudaMemcpy(dev_frame, frame_gray.ptr<float>(), frameSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(templ_const, templ_gray.ptr<float>(), templPixels * sizeof(float), 0, cudaMemcpyHostToDevice);
    dim3 block(32, 8);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
    nccKernelConst<<<grid, block>>>(dev_frame, frameW, frameH, templW, templH, templMean, templStd, dev_out, outW, outH);
    cudaDeviceSynchronize();
    cudaMemcpy(ncc_map.ptr<float>(), dev_out, outSize, cudaMemcpyDeviceToHost);
    cudaFree(dev_frame);
    cudaFree(dev_out);
}

void ncc_match_const_tiled(const cv::Mat& frame_gray, const cv::Mat& templ_gray, cv::Mat& ncc_map)
{
    int frameW = frame_gray.cols;
    int frameH = frame_gray.rows;
    int templW = templ_gray.cols;
    int templH = templ_gray.rows;
    int outW = frameW - templW + 1;
    int outH = frameH - templH + 1;
    int templPixels = templW * templH;
    CV_Assert(templPixels <= MAX_TEMPL_PIXELS);
    ncc_map.create(outH, outW, CV_32FC1);
    cv::Scalar mean, stddev;
    cv::meanStdDev(templ_gray, mean, stddev);
    float templMean = (float)(mean[0]);
    float templStd  = (float)(stddev[0] + 1e-6f);
    size_t frameSize = (size_t)(frameW) * frameH * sizeof(float);
    size_t outSize = (size_t)(outW) * outH * sizeof(float);
    float *dev_frame = NULL, *dev_out = NULL;
    cudaMalloc(&dev_frame, frameSize);
    cudaMalloc(&dev_out,   outSize);
    cudaMemcpy(dev_frame, frame_gray.ptr<float>(), frameSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(templ_const, templ_gray.ptr<float>(), templPixels * sizeof(float), 0, cudaMemcpyHostToDevice);
    dim3 block(32, 8);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
    int tileW = block.x + templW - 1;
    int tileH = block.y + templH - 1;
    size_t tileSize = (size_t)(tileW) * tileH * sizeof(float);
    nccKernelConstTiled<<<grid, block, tileSize>>>(dev_frame, frameW, frameH, templW, templH, templMean, templStd, dev_out, outW, outH);
    cudaDeviceSynchronize();
    cudaMemcpy(ncc_map.ptr<float>(), dev_out, outSize, cudaMemcpyDeviceToHost);
    cudaFree(dev_frame);
    cudaFree(dev_out);
}