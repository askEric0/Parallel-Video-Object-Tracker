#pragma once
#include <opencv2/opencv.hpp>

inline cv::Mat to_gray(const cv::Mat& bgr) {
    cv::Mat gray, gray_f32;
    if (bgr.channels() == 3) {
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = bgr;
    }
    gray.convertTo(gray_f32, CV_32F, 1.0f / 255.0f);
    return gray_f32;
}
