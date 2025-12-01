#include "baseline_kernel.hpp"

namespace baseline {

// Simple CPU baseline using OpenCV's normalized cross-correlation.
// frame_gray_f32 and templ_gray_f32 are single-channel CV_32F images in [0, 1].
void ncc_match_cpu(const cv::Mat& frame_gray_f32,
                   const cv::Mat& templ_gray_f32,
                   cv::Mat& ncc_map)
{
    CV_Assert(frame_gray_f32.type() == CV_32FC1);
    CV_Assert(templ_gray_f32.type() == CV_32FC1);
    CV_Assert(frame_gray_f32.cols >= templ_gray_f32.cols);
    CV_Assert(frame_gray_f32.rows >= templ_gray_f32.rows);

    // Use OpenCV's optimized implementation of NCC.
    // zero‑mean normalized cross‑correlation
    cv::matchTemplate(frame_gray_f32, templ_gray_f32, ncc_map,
                      cv::TM_CCOEFF_NORMED);
}

} // namespace baseline


