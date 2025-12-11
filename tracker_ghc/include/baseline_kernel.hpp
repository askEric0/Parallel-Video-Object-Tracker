
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>


void ncc_match_naive_cuda(const cv::Mat& frame_gray, const cv::Mat& templ_gray, cv::Mat& ncc_map);
void ncc_match_shared_cuda(const cv::Mat& frame_gray, const cv::Mat& templ_gray, cv::Mat& ncc_map);
void ncc_match_cpu(const cv::Mat& frame_gray, const cv::Mat& templ_gray, cv::Mat& ncc_map);
void ncc_match_naive_cuda_batched(const std::vector<cv::Mat>& frames_gray, const cv::Mat& templ_gray, std::vector<cv::Mat>& ncc_maps);
void ncc_match_const(const cv::Mat& frame_gray, const cv::Mat& templ_gray, cv::Mat& ncc_map);
void ncc_match_const_tiled(const cv::Mat& frame_gray, const cv::Mat& templ_gray, cv::Mat& ncc_map);
