#include "baseline_kernel.hpp"

namespace baseline {

void ncc_match_cpu(const cv::Mat& frame_gray_f32, const cv::Mat& templ_gray_f32, cv::Mat& ncc_map)
{
    CV_Assert(frame_gray_f32.type() == CV_32FC1);
    CV_Assert(templ_gray_f32.type() == CV_32FC1);
    CV_Assert(frame_gray_f32.cols >= templ_gray_f32.cols);
    CV_Assert(frame_gray_f32.rows >= templ_gray_f32.rows);

    cv::matchTemplate(frame_gray_f32, templ_gray_f32, ncc_map, cv::TM_CCOEFF_NORMED);
}

} // namespace baseline


