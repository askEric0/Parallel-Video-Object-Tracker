#include <opencv2/opencv.hpp>
#include <iostream>
#include "baseline_kernel.hpp"
#include "utils.hpp"

static const std::string INPUT_VIDEO = "../data/car.mp4";

static const std::string NCC_MODE = "naive";  

// Batch size
static const int BATCH_SIZE = 4;

// Search window
static const int SEARCH_RADIUS_X = 80;
static const int SEARCH_RADIUS_Y = 80;

// Confidence 
static const double NCC_MIN_CONFIDENCE    = 0.40;
static const double NCC_STRONG_CONFIDENCE = 0.70;
static const double TEMPLATE_UPDATE_LR    = 0.10;


int main(int argc, char** argv)
{

    std::string mode = NCC_MODE; 
    int batch = BATCH_SIZE;

    for (int i = 1; i < argc; ++i) {
        if (i == 2) break;
        std::string arg = argv[i];

        if (arg == "--cpu") mode = "cpu";
        else if (arg == "--shared") mode = "shared";
        else if (arg == "--const") mode = "const";
        else if (arg == "--const_tiled") mode = "const_tiled";
        else if (arg.rfind("--batch=", 0) == 0) {
            mode = "batch";
            batch = std::max(1, std::atoi(arg.substr(8).c_str()));
        }
    }

    std::cout << "--------\n";
    std::cout << "NCC Tracker Starting\n";
    std::cout << "Input video : " << INPUT_VIDEO << "\n";
    std::cout << "Mode        : " << mode << "\n";
    if (mode == "batch")
        std::cout << "Batch size  : " << batch << "\n";
    std::cout << "--------\n\n";

    // Load input video
    cv::VideoCapture cap(INPUT_VIDEO);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video.\n";
        return -1;
    }

    cv::Mat frame;
    cap >> frame;
    if (frame.empty()) return -1;

    // ROI
    cv::Rect bbox = cv::selectROI("Select Object", frame, false);
    cv::destroyWindow("Select Object");

    if (bbox.width == 0 || bbox.height == 0) {
        std::cerr << " No ROI selected.\n";
        return -1;
    }
    cv::Mat frame_gray_f32 = toGrayF32(frame);
    cv::Mat templ_gray_f32 = frame_gray_f32(bbox).clone();

    double fps_video = cap.get(cv::CAP_PROP_FPS);
    if (fps_video <= 1) fps_video = 30;

    cv::VideoWriter writer("../output/output_tracker.mp4", cv::VideoWriter::fourcc('m','p','4','v'),
                           fps_video, cv::Size(frame.cols, frame.rows));

    if (!writer.isOpened()) {
        std::cerr << " Cannot open video writer.\n";
        return -1;
    }

    std::vector<cv::Mat> batch_frames;
    std::vector<cv::Mat> batch_ncc;
    if (mode == "batch") batch_frames.reserve(batch);

    int frame_count = 0;
    double t_start = (double)cv::getTickCount();

    // Loop over video frames
    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        frame_gray_f32 = toGrayF32(frame);
        cv::Mat ncc_map;

        if (mode == "cpu") {
            baseline::ncc_match_cpu(frame_gray_f32, templ_gray_f32, ncc_map);
        }
        else if (mode == "shared") {
            baseline::ncc_match_shared_cuda(frame_gray_f32, templ_gray_f32, ncc_map);
        }
        else if (mode == "const") {
            baseline::ncc_match_const(frame_gray_f32, templ_gray_f32, ncc_map);
        }
        else if (mode == "const_tiled") {
            baseline::ncc_match_const_tiled(frame_gray_f32, templ_gray_f32, ncc_map);
        }
        else if (mode == "batch") {
            batch_frames.push_back(frame_gray_f32.clone());

            if ((int)batch_frames.size() < batch) {
                cv::rectangle(frame, bbox, {0,255,0}, 2);
                writer.write(frame);
                frame_count++;
                continue;
            }

            baseline::ncc_match_naive_cuda_batched(batch_frames,templ_gray_f32,batch_ncc);

            ncc_map = batch_ncc.back();
            batch_frames.clear();
            batch_ncc.clear();
        }
        else {
            baseline::ncc_match_naive_cuda(frame_gray_f32, templ_gray_f32, ncc_map);
        }

        int cx = bbox.x + bbox.width / 2;
        int cy = bbox.y + bbox.height / 2;

        int outW = ncc_map.cols;
        int outH = ncc_map.rows;

        int minTx = std::max(0, cx - SEARCH_RADIUS_X - bbox.width / 2);
        int maxTx = std::min(outW - 1, cx + SEARCH_RADIUS_X - bbox.width / 2);
        int minTy = std::max(0, cy - SEARCH_RADIUS_Y - bbox.height / 2);
        int maxTy = std::min(outH - 1, cy + SEARCH_RADIUS_Y - bbox.height / 2);

        cv::Rect roi(minTx, minTy, maxTx - minTx + 1, maxTy - minTy + 1);
        cv::Point bestLoc;
        double bestVal;

        cv::minMaxLoc(ncc_map(roi), nullptr, &bestVal, nullptr, &bestLoc);
        bestLoc += roi.tl();

        if (bestVal >= NCC_MIN_CONFIDENCE) {
            bbox.x = bestLoc.x;
            bbox.y = bestLoc.y;

            if (bestVal >= NCC_STRONG_CONFIDENCE) {
                cv::Mat new_patch = frame_gray_f32(bbox).clone();
                cv::addWeighted(templ_gray_f32, 1 - TEMPLATE_UPDATE_LR, new_patch, TEMPLATE_UPDATE_LR,  0.0, templ_gray_f32);
            }
        }

        cv::rectangle(frame, bbox, {0,255,0}, 2);
        writer.write(frame);
        frame_count++;
    }

    double t_end = (double)cv::getTickCount();
    double elapsed = (t_end - t_start) / cv::getTickFrequency();
    double fps = frame_count / elapsed;

    std::cout << "\n--------\n";
    std::cout << " Tracking Complete\n";
    std::cout << " Mode       : " << mode << "\n";
    std::cout << " Frames     : " << frame_count << "\n";
    std::cout << " Time (sec) : " << elapsed << "\n";
    std::cout << " FPS        : " << fps << "\n";
    std::cout << "--------\n";

    return 0;
}
