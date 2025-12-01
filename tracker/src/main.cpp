#include <opencv2/opencv.hpp>
#include <iostream>
#include "baseline_kernel.hpp"
#include "utils.hpp"

// ------------------------------------------------------------
// High-level description
// ------------------------------------------------------------
// This file implements the end-to-end tracker:
//   - Load video (OpenCV).
//   - Let user scrub to a frame and draw an initial bounding box.
//   - Convert first frame to grayscale float and extract template.
//   - For each subsequent frame:
//        * Convert to grayscale float.
//        * Run NCC (CPU or CUDA) inside a local search window
//          around the previous bounding box.
//        * Update bbox, draw it, show FPS and optional heatmap.
//        * Save annotated frames to an output MP4.
//
// It is intentionally "baseline": a single fixed template and
// plain NCC matching, mainly for correctness + performance study.

// Search window radius around current bbox center (in pixels).
// We restrict NCC search to this window in the NCC map to reduce drift
// to background and to prefer staying near the previous location.
static const int SEARCH_RADIUS_X = 80;
static const int SEARCH_RADIUS_Y = 80;

// Minimum acceptable NCC score; below this we keep the previous bbox
// instead of jumping to a low-confidence match.
static const double NCC_MIN_CONFIDENCE = 0.5;

// If NCC score is this high, we treat the match as very reliable and use
// it to slowly update (adapt) the template to handle appearance / scale
// changes over time.
static const double NCC_STRONG_CONFIDENCE = 0.7;

// Template update learning rate: how much of the new patch we blend into
// the existing template at each strong, confident match.
static const double TEMPLATE_UPDATE_LR = 0.1;

// Utility mode forward declarations
static int run_interactive_tracker(const std::string& video_path,
                                   bool use_cuda);
static int run_cpu_cuda_check(const std::string& video_path);
static int run_benchmark(const std::string& video_path,
                         bool use_cuda,
                         bool use_shared_kernel,
                         int batch_size);

int main(int argc, char** argv) {
    // ---------------------------------------------------------------------
    // Parse command-line arguments
    //   argv[1] : input video path      (default: "data/test.mp4")
    //   other args:
    //      --cpu        : use CPU baseline (OpenCV matchTemplate)
    //      --check      : run CPU vs CUDA numeric check and exit
    //      --bench      : run headless benchmark and exit
    //      --shared     : use shared-memory kernel (CUDA only, for bench)
    //      --batch=N    : batch size for benchmark (default 4)
    // ---------------------------------------------------------------------
    std::string video_path = (argc > 1) ? argv[1] : "../data/car.mp4";

    bool use_cuda = true;
    bool run_check_mode = false;
    bool run_bench_mode = false;
    bool use_shared_kernel = false;
    int batch_size = 1;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu")          use_cuda = false;
        else if (arg == "--check")  run_check_mode = true;
        else if (arg == "--bench")  run_bench_mode = true;
        else if (arg == "--shared") use_shared_kernel = true;
        else if (arg.rfind("--batch=", 0) == 0) {
            batch_size = std::max(1, std::atoi(arg.substr(8).c_str()));
        }
    }

    if (run_check_mode) {
        return run_cpu_cuda_check(video_path);
    }
    if (run_bench_mode) {
        return run_benchmark(video_path, use_cuda, use_shared_kernel, batch_size);
    }

    return run_interactive_tracker(video_path, use_cuda);
}

// ---------------------------------------------------------------------
// Interactive tracker mode (GUI), used for demos.
// ---------------------------------------------------------------------
static int run_interactive_tracker(const std::string& video_path,
                                   bool use_cuda)
{
    // -------------------- Load Video --------------------
    cv::VideoCapture cap;
    cv::Mat firstFrame;

    auto t0 = std::chrono::high_resolution_clock::now();
    cap.open(video_path);
    if (!cap.isOpened()) {
        std::cerr << "❌ Cannot open video: " << video_path << "\n";
        return -1;
    }

    cap >> firstFrame;
    if (firstFrame.empty()) {
        std::cerr << "❌ Empty first frame\n";
        return -1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double t_open = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // -------------------- ROI Selection --------------------
    auto t2 = std::chrono::high_resolution_clock::now();
    cv::Rect curr_bbox = cv::selectROI("Select Object", firstFrame, false);
    cv::destroyWindow("Select Object");
    auto t3 = std::chrono::high_resolution_clock::now();
    double t_roi = std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (curr_bbox.width == 0 || curr_bbox.height == 0) {
        std::cerr << "❌ No ROI selected\n";
        return -1;
    }

    // -------------------- Template Init --------------------
    cv::Mat frame_gray_f32 = toGrayF32(firstFrame);
    cv::Mat templ_gray_f32 = frame_gray_f32(curr_bbox).clone();

    // -------------------- Output Writer --------------------
    cv::VideoWriter writer;
    double fps_video = cap.get(cv::CAP_PROP_FPS);
    if (fps_video <= 1.0) fps_video = 30.0;

    auto t4 = std::chrono::high_resolution_clock::now();
    writer.open("../output/output_tracker.mp4",
                cv::VideoWriter::fourcc('m','p','4','v'),
                fps_video,
                cv::Size(firstFrame.cols, firstFrame.rows));
    auto t5 = std::chrono::high_resolution_clock::now();
    double t_writer = std::chrono::duration<double, std::milli>(t5 - t4).count();

    if (!writer.isOpened()) {
        std::cerr << "❌ Cannot open output video writer\n";
        return -1;
    }

    // -------------------- Setup Report --------------------
    std::cout << "\n=========== NCC SETUP PROFILING ===========\n";
    std::cout << "Video open time  : " << t_open   << " ms\n";
    std::cout << "ROI select time  : " << t_roi    << " ms\n";
    std::cout << "Writer init      : " << t_writer << " ms\n";
    std::cout << "==========================================\n";

    // -------------------- Tracking Loop --------------------
    cv::Mat frame;
    int frame_count = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        frame_gray_f32 = toGrayF32(frame);

        // ---------- NCC Computation ----------
        cv::Mat ncc_map;
        if (use_cuda)
            baseline::ncc_match_naive_cuda(frame_gray_f32, templ_gray_f32, ncc_map);
        else
            baseline::ncc_match_cpu(frame_gray_f32, templ_gray_f32, ncc_map);

        // ---------- ✅ RESTORED LOCAL SEARCH WINDOW ----------
        int cx = curr_bbox.x + curr_bbox.width  / 2;
        int cy = curr_bbox.y + curr_bbox.height / 2;

        int outW = ncc_map.cols;
        int outH = ncc_map.rows;

        int minTx = std::max(0, cx - SEARCH_RADIUS_X - templ_gray_f32.cols / 2);
        int maxTx = std::min(outW - 1, cx + SEARCH_RADIUS_X - templ_gray_f32.cols / 2);
        int minTy = std::max(0, cy - SEARCH_RADIUS_Y - templ_gray_f32.rows / 2);
        int maxTy = std::min(outH - 1, cy + SEARCH_RADIUS_Y - templ_gray_f32.rows / 2);

        int searchW = maxTx - minTx + 1;
        int searchH = maxTy - minTy + 1;

        cv::Point bestLoc;
        double bestVal = -1.0;

        if (searchW > 0 && searchH > 0) {
            cv::Rect roi(minTx, minTy, searchW, searchH);
            cv::Mat ncc_roi = ncc_map(roi);

            double minVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(ncc_roi, &minVal, &bestVal, &minLoc, &maxLoc);

            bestLoc = maxLoc + roi.tl();   // ✅ correct global location
        } else {
            double minVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(ncc_map, &minVal, &bestVal, &minLoc, &maxLoc);
            bestLoc = maxLoc;
        }

        // ---------- Update BBox ----------
        if (bestVal >= NCC_MIN_CONFIDENCE) {
            curr_bbox.x = bestLoc.x;
            curr_bbox.y = bestLoc.y;
            curr_bbox.width  = templ_gray_f32.cols;
            curr_bbox.height = templ_gray_f32.rows;

            // ---------- ✅ RESTORED TEMPLATE UPDATE ----------
            if (bestVal >= NCC_STRONG_CONFIDENCE) {
                cv::Mat new_patch = frame_gray_f32(curr_bbox).clone();
                cv::addWeighted(templ_gray_f32, 1.0 - TEMPLATE_UPDATE_LR,
                                new_patch, TEMPLATE_UPDATE_LR,
                                0.0, templ_gray_f32);
            }
        }

        // ---------- Draw ----------
        cv::rectangle(frame, curr_bbox, {0,255,0}, 2);
        cv::putText(frame,
                    use_cuda ? "NCC CUDA" : "NCC CPU",
                    {20,30},
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8, {0,255,0}, 2);

        // ---------- Write ----------
        writer.write(frame);

        // // ---------- ✅ RESTORED SCALED DISPLAY ----------
        // cv::Mat display_frame;
        // cv::resize(frame, display_frame, {}, 0.5, 0.5, cv::INTER_AREA);
        // cv::imshow("Tracking", display_frame);

        // if (cv::waitKey(1) == 27) break;

        frame_count++;
    }

    // -------------------- Clean Unload --------------------
    writer.release();
    cap.release();
    cv::destroyAllWindows();

    std::cout << "✅ NCC Tracking finished. Frames: " << frame_count << "\n";
    return 0;
}



// ---------------------------------------------------------------------
// Mode: numeric check CPU vs CUDA on a single frame.
// ---------------------------------------------------------------------
static int run_cpu_cuda_check(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return -1;
    }

    cv::Mat frame;
    if (!cap.read(frame)) {
        std::cerr << "Cannot read first frame" << std::endl;
        return -1;
    }

    cv::Rect roi = cv::selectROI("Select ROI (for check)", frame, false, false);
    if (roi.width == 0 || roi.height == 0) {
        std::cerr << "No ROI selected" << std::endl;
        return -1;
    }
    cv::destroyWindow("Select ROI (for check)");

    cv::Mat frame_gray_f32 = toGrayF32(frame);
    cv::Mat templ_gray_f32 = frame_gray_f32(roi).clone();

    cv::Mat ncc_cpu, ncc_cuda;
    baseline::ncc_match_cpu(frame_gray_f32, templ_gray_f32, ncc_cpu);
    baseline::ncc_match_naive_cuda(frame_gray_f32, templ_gray_f32, ncc_cuda);

    cv::Mat diff;
    cv::absdiff(ncc_cpu, ncc_cuda, diff);
    double minDiff, maxDiff;
    cv::minMaxLoc(diff, &minDiff, &maxDiff);

    std::cout << "CPU vs CUDA naive NCC max abs diff: " << maxDiff << std::endl;
    return 0;
}

// ---------------------------------------------------------------------
// Headless benchmark mode: no GUI, just timing.
// ---------------------------------------------------------------------
static int run_benchmark(const std::string& video_path,
                         bool use_cuda,
                         bool use_shared_kernel,
                         int batch_size) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return -1;
    }

    cv::Mat frame;
    if (!cap.read(frame)) {
        std::cerr << "Cannot read first frame" << std::endl;
        return -1;
    }

    cv::Mat frame_gray_f32 = toGrayF32(frame);

    // Use a fixed central 64x64 (or smaller) patch as template to avoid GUI.
    int templW = std::min(64, frame_gray_f32.cols - 1);
    int templH = std::min(64, frame_gray_f32.rows - 1);
    int x0 = (frame_gray_f32.cols - templW) / 2;
    int y0 = (frame_gray_f32.rows - templH) / 2;
    cv::Mat templ_gray_f32 = frame_gray_f32(cv::Rect(x0, y0, templW, templH)).clone();

    int total_frames = 0;
    double t_start = static_cast<double>(cv::getTickCount());

    if (!use_cuda) {
        // CPU benchmark: process frames one by one.
        do {
            cv::Mat ncc;
            baseline::ncc_match_cpu(frame_gray_f32, templ_gray_f32, ncc);
            total_frames++;
        } while (cap.read(frame) && (frame_gray_f32 = toGrayF32(frame), true));
    } else if (use_shared_kernel) {
        // CUDA shared-memory kernel, single frame per launch.
        do {
            cv::Mat ncc;
            baseline::ncc_match_shared_cuda(frame_gray_f32, templ_gray_f32, ncc);
            total_frames++;
        } while (cap.read(frame) && (frame_gray_f32 = toGrayF32(frame), true));
    } else {
        // CUDA naive batched benchmark.
        std::vector<cv::Mat> batch_frames;
        batch_frames.reserve(batch_size);

        while (true) {
            batch_frames.push_back(frame_gray_f32.clone());
            total_frames++;

            bool no_more = !cap.read(frame);
            if ((int)batch_frames.size() == batch_size || no_more) {
                std::vector<cv::Mat> ncc_maps;
                baseline::ncc_match_naive_cuda_batched(batch_frames,
                                                       templ_gray_f32,
                                                       ncc_maps);
                batch_frames.clear();
                if (no_more) break;
            }

            frame_gray_f32 = toGrayF32(frame);
        }
    }

    double t_end = static_cast<double>(cv::getTickCount());
    double elapsed = (t_end - t_start) / cv::getTickFrequency();
    double fps = (elapsed > 0.0) ? total_frames / elapsed : 0.0;

    std::cout << "Benchmark result (" << (use_cuda ? "CUDA" : "CPU")
              << (use_cuda && use_shared_kernel ? ", shared" : use_cuda ? ", naive batched" : "")
              << "): frames=" << total_frames
              << ", time=" << elapsed << " s"
              << ", FPS=" << fps << std::endl;

    return 0;
}
