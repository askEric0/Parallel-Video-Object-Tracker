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
// A moderate radius works well for car.mp4: small enough to avoid
// jumping to distant distractors, large enough to handle motion.
static const int SEARCH_RADIUS_X = 60;
static const int SEARCH_RADIUS_Y = 60;

// Minimum acceptable NCC score; below this we keep the previous bbox
// instead of jumping to a low-confidence match.
// Slightly lower threshold so we don't freeze too easily, but still
// ignore clearly bad matches.
static const double NCC_MIN_CONFIDENCE = 0.4;

// We disable template adaptation for robustness: keep the original
// template fixed so we don't accidentally "learn" the bush/road.
static const double NCC_STRONG_CONFIDENCE = 1.0;
static const double TEMPLATE_UPDATE_LR = 0.0;

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
    std::string video_path = (argc > 1) ? argv[1] : "data/test.mp4";

    bool use_cuda = true;
    bool run_check_mode = false;
    bool run_bench_mode = false;
    bool use_shared_kernel = false;
    int batch_size = 4;

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
static int run_interactive_tracker(const std::string& video_path, bool use_cuda) {

    // ---------------------------------------------------------------------
    // Open video file
    // ---------------------------------------------------------------------
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return -1;
    }

    // If there is no DISPLAY (e.g., running on a headless server), the
    // interactive GUI-based tracker cannot show windows. In that case,
    // fall back to the headless benchmark mode so the program can still
    // run without user interaction.
    const char* display = std::getenv("DISPLAY");
    if (display == nullptr || std::string(display).empty()) {
        std::cout << "[Headless] DISPLAY not set; running benchmark mode instead "
                     "of interactive GUI tracker.\n";
        // Use a reasonable default batch size and no shared-kernel by default.
        bool use_shared_kernel = false;
        int batch_size = 4;
        return run_benchmark(video_path, use_cuda, use_shared_kernel, batch_size);
    }

    // ---------------------------------------------------------------------
    // Let the user choose WHICH frame to start from.
    // We show frames in a preview window; when the user presses SPACE or ENTER,
    // we freeze on that frame and ask them to draw the ROI. This allows
    // skipping initial frames that do not contain the target object.
    // ---------------------------------------------------------------------
    cv::Mat frame;
    cv::namedWindow("Frame Preview", cv::WINDOW_NORMAL);
    std::cout << "Use the preview window to pick a frame that contains the target object.\n"
              << "Press SPACE or ENTER to select the current frame, or Q/ESC to quit.\n";

    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Reached end of video before ROI selection." << std::endl;
            return -1;
        }

        cv::imshow("Frame Preview", frame);
        int key = cv::waitKey(30);

        if (key == 27 || key == 'q') { // ESC or 'q'
            std::cout << "ROI selection cancelled by user." << std::endl;
            return 0;
        }
        if (key == ' ' || key == '\r' || key == '\n') { // SPACE or ENTER
            break;
        }
    }
    cv::destroyWindow("Frame Preview");

    // Select initial ROI on the chosen frame
    cv::Rect roi = cv::selectROI("Select ROI", frame, false, false);
    if (roi.width == 0 || roi.height == 0) {
        std::cerr << "No ROI selected" << std::endl;
        return -1;
    }
    cv::destroyWindow("Select ROI");

    // Prepare template from first frame.
    // First convert to grayscale float.
    cv::Mat frame_gray_f32 = toGrayF32(frame);

    // Use the entire user-selected ROI as the template, so the template
    // window matches exactly what you drew (e.g., the whole balloon).
    cv::Mat templ_gray_f32 = frame_gray_f32(roi).clone();

    // Show the template so you can visually confirm what is being tracked.
    cv::Mat templ_vis;
    templ_gray_f32.convertTo(templ_vis, CV_8U, 255.0);
    cv::namedWindow("Template", cv::WINDOW_NORMAL);
    cv::imshow("Template", templ_vis);
    cv::waitKey(1);

    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
    cv::namedWindow("NCC Heatmap", cv::WINDOW_NORMAL);

    // Current estimate of object location in full-frame coordinates.
    cv::Rect curr_bbox = roi;

    // Timestamp of previous frame (for FPS computation).
    int64 last_tick = cv::getTickCount();

    // ---------------------------------------------------------------------
    // Main tracking loop: process frames until the video ends or user quits.
    // ---------------------------------------------------------------------
    while (true) {
        if (!cap.read(frame)) {
            break; // end of video
        }

        frame_gray_f32 = toGrayF32(frame);

        // -------------------------------------------------------------
        // NCC-based tracking:
        //   - Compute NCC over the full frame.
        //   - But when choosing the best location, restrict to a local
        //     window in the NCC map around the previous bbox center.
        //   - Optionally reject very low-confidence matches.
        // -------------------------------------------------------------
        int frameW = frame_gray_f32.cols;
        int frameH = frame_gray_f32.rows;
        int templW = templ_gray_f32.cols;
        int templH = templ_gray_f32.rows;

        // Run either CPU or CUDA NCC over the full frame
        cv::Mat ncc_map;
        if (use_cuda) {
            baseline::ncc_match_naive_cuda(frame_gray_f32, templ_gray_f32, ncc_map);
        } else {
            baseline::ncc_match_cpu(frame_gray_f32, templ_gray_f32, ncc_map);
        }

        // Compute a local search window in NCC space around the current
        // bbox center. The NCC map is indexed by the template's top-left
        // position, so we convert from center coordinates.
        int cx = curr_bbox.x + curr_bbox.width  / 2;
        int cy = curr_bbox.y + curr_bbox.height / 2;

        int outW = ncc_map.cols;
        int outH = ncc_map.rows;

        int minTx = std::max(0, cx - SEARCH_RADIUS_X - templW / 2);
        int maxTx = std::min(outW - 1, cx + SEARCH_RADIUS_X - templW / 2);
        int minTy = std::max(0, cy - SEARCH_RADIUS_Y - templH / 2);
        int maxTy = std::min(outH - 1, cy + SEARCH_RADIUS_Y - templH / 2);

        int searchW = maxTx - minTx + 1;
        int searchH = maxTy - minTy + 1;

        cv::Point bestLoc;
        double bestVal = -1.0;

        if (searchW > 0 && searchH > 0) {
            cv::Rect ncc_search_roi(minTx, minTy, searchW, searchH);
            cv::Mat ncc_roi = ncc_map(ncc_search_roi);

            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(ncc_roi, &minVal, &maxVal, &minLoc, &maxLoc);

            bestVal = maxVal;
            bestLoc = cv::Point(maxLoc.x + minTx, maxLoc.y + minTy);
        } else {
            // Fallback: if the search window collapses (e.g., near borders),
            // use the global best match.
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(ncc_map, &minVal, &maxVal, &minLoc, &maxLoc);
            bestVal = maxVal;
            bestLoc = maxLoc;
        }

        // If the best NCC score is too low, keep the previous bbox instead
        // of jumping to a dubious location. We no longer adapt the template;
        // we simply move the box to the local NCC peak when confidence is
        // good enough.
        if (bestVal >= NCC_MIN_CONFIDENCE) {
            curr_bbox.x = bestLoc.x;
            curr_bbox.y = bestLoc.y;
            curr_bbox.width  = templW;
            curr_bbox.height = templH;
        }

        // Draw bbox
        cv::rectangle(frame, curr_bbox, cv::Scalar(0, 255, 0), 2);

        // FPS calculation
        int64 now_tick = cv::getTickCount();
        double dt = (now_tick - last_tick) / cv::getTickFrequency();
        last_tick = now_tick;
        double fps = (dt > 0.0) ? (1.0 / dt) : 0.0;

        std::string fps_text = cv::format("FPS: %.1f", fps);
        cv::putText(frame, fps_text, cv::Point(20, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        // Write full-resolution frame to output video if available
        // if (writer.isOpened()) {
        //     writer.write(frame);
        // }

        // For display, show a downscaled copy so the whole frame fits
        // in a smaller window (helps over X11 / remote desktop).
        cv::Mat display_frame;
        const double display_scale = 0.5; // adjust if you want bigger/smaller
        cv::resize(frame, display_frame, cv::Size(), display_scale, display_scale,
                   cv::INTER_AREA);
        cv::imshow("Tracking", display_frame);

        // NCC heatmap visualization
        cv::Mat ncc_norm, ncc_color;
        cv::normalize(ncc_map, ncc_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(ncc_norm, ncc_color, cv::COLORMAP_JET);
        cv::imshow("NCC Heatmap", ncc_color);

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            break;
        }
    }

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
