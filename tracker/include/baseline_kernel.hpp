
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace baseline {

// ------------------------------------------------------------
// Single-frame NCC APIs (used by interactive tracker and checks)
// ------------------------------------------------------------
void ncc_match_naive_cuda(const cv::Mat& frame_gray_f32,
                          const cv::Mat& templ_gray_f32,
                          cv::Mat& ncc_map);

void ncc_match_shared_cuda(const cv::Mat& frame_gray_f32,
                           const cv::Mat& templ_gray_f32,
                           cv::Mat& ncc_map);

// CPU baseline implementation using OpenCV's normalized cross-correlation.
// This is used for correctness checking and speedup comparison.
void ncc_match_cpu(const cv::Mat& frame_gray_f32,
                   const cv::Mat& templ_gray_f32,
                   cv::Mat& ncc_map);

// ------------------------------------------------------------
// Batched NCC API for benchmarking throughput
//   - frames_gray_f32: vector of grayscale CV_32FC1 frames
//   - templ_gray_f32 : grayscale CV_32FC1 template
//   - ncc_maps       : output NCC maps (one per frame)
// ------------------------------------------------------------
void ncc_match_naive_cuda_batched(const std::vector<cv::Mat>& frames_gray_f32,
                                  const cv::Mat& templ_gray_f32,
                                  std::vector<cv::Mat>& ncc_maps);

void ncc_match_const(const cv::Mat& frame_gray_f32,
                               const cv::Mat& templ_gray_f32,
                               cv::Mat& ncc_map) ;                           
} // namespace baseline
