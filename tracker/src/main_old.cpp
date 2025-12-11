#include <opencv2/opencv.hpp>
#include <iostream>
#include "baseline_kernel.hpp"
#include "utils.hpp"

// Local search window size around the current bounding box center
static const int SEARCH_RADIUS_X = 60; // 80
static const int SEARCH_RADIUS_Y = 60; // 80
// Default batch size
static const int BATCH_SIZE = 4;
// Minimum acceptable NCC score. Keep the previous bbox if the score is below this threshold.
static const double NCC_MIN_CONFIDENCE = 0.4;
// Strong confidence threshold. Update the template if the score is above this threshold.
static const double NCC_STRONG_CONFIDENCE = 0.7; // 1.0
// Learning rate for template adaptation.
static const double TEMPLATE_UPDATE_LR = 0.1; // 0.0

static int demo_tracker(const std::string& video_path, std::string mode);
static int record_tracker(const std::string& video_path, std::string mode, int batch_size, const std::string& output_path);

static std::string generate_output_path(const std::string& video_path) {
    size_t last_slash = video_path.find_last_of("/\\");
    size_t last_dot = video_path.find_last_of(".");
    
    std::string dir = "";
    std::string base = video_path;
    std::string ext = ".mp4";
    
    if (last_slash != std::string::npos) {
        dir = video_path.substr(0, last_slash + 1);
        base = video_path.substr(last_slash + 1);
    }
    
    if (last_dot != std::string::npos && last_dot > last_slash) {
        ext = video_path.substr(last_dot);
        base = base.substr(0, base.find_last_of("."));
    }
    
    return dir + base + "_tracked" + ext;
}

int main(int argc, char** argv) {
    std::string video_path = (argc > 1) ? argv[1] : "data/car.mp4";
    std::string mode = "cuda";
    bool record = false;
    int batch_size = 0;
    std::string output_path = generate_output_path(video_path);

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") mode = "cpu";
        else if (arg == "--shared")  mode = "shared";
        else if (arg == "--const") mode = "const";
        else if (arg == "--const_tiled") mode = "const_tiled";
        else if (arg == "--record")  record = true;
        else if (arg.rfind("--batch=", 0) == 0) {
            mode = "batch";
            batch_size = std::max(1, std::atoi(arg.substr(8).c_str()));
        }
    }

    if (record) return record_tracker(video_path, mode, batch_size, output_path);
    return demo_tracker(video_path, mode);
}

static int demo_tracker(const std::string& video_path, std::string mode) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return -1;
    }
    cv::Mat frame;
    cv::namedWindow("Frame Preview", cv::WINDOW_NORMAL);
    std::cout << "Use the preview window to pick a frame that contains the target object.\n"
              << "Press ENTER to select the current frame. Press ESC to quit.\n";
    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Reached End of Video." << std::endl;
            return -1;
        }
        cv::imshow("Frame Preview", frame);
        int key = cv::waitKey(30);

        if (key == 27) { 
            std::cout << "Template selection cancelled by user." << std::endl;
            return 0;
        }
        if (key == '\r' || key == '\n') break;
    }
    cv::destroyWindow("Frame Preview");
    cv::Rect roi = cv::selectROI("Select Template", frame, false, false);
    if (roi.width == 0 || roi.height == 0) {
        std::cerr << "No template selected" << std::endl;
        return -1;
    }
    cv::destroyWindow("Select Template");
    cv::Mat frame_gray = to_gray(frame);
    cv::Mat templ_gray = frame_gray(roi).clone();
    // Show the template so you can visually confirm what is being tracked.
    cv::Mat roi_visual;
    templ_gray.convertTo(roi_visual, CV_8U, 255.0);
    cv::namedWindow("Template", cv::WINDOW_NORMAL);
    cv::imshow("Template", roi_visual);
    cv::waitKey(1);
    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
    // Current estimate of object location in full-frame coordinates.
    cv::Rect curr_bbox = roi;
    // Timestamp of previous frame for FPS computation.
    int64 last_tick = cv::getTickCount();
    // Overall timing for interactive tracking.
    int64 t_start = cv::getTickCount();
    int total_frames = 0;

    while (true) {
        // End of video.    
        if (!cap.read(frame)) break;
        // Convert the frame to grayscale float.
        frame_gray = to_gray(frame);
        // Get the dimensions of the frame and template.
        int frameW = frame_gray.cols;
        int frameH = frame_gray.rows;
        int templW = templ_gray.cols;
        int templH = templ_gray.rows;
        // Compute the NCC map over the full frame.
        cv::Mat ncc_map;
        if (mode == "cpu") {
            cv::matchTemplate(frame_gray, templ_gray, ncc_map, cv::TM_CCOEFF_NORMED);
        }
        else if (mode == "shared") {
            ncc_match_shared_cuda(frame_gray, templ_gray, ncc_map);
        }
        else if (mode == "const") {
            ncc_match_const(frame_gray, templ_gray, ncc_map);
        }
        else if (mode == "const_tiled") {
            ncc_match_const_tiled(frame_gray, templ_gray, ncc_map);
        }
        else {
            ncc_match_naive_cuda(frame_gray, templ_gray, ncc_map);
        }
        // Get the center of the current bounding box.
        int cx = curr_bbox.x + curr_bbox.width  / 2;
        int cy = curr_bbox.y + curr_bbox.height / 2;
        // Get the dimensions of the NCC map.
        int outW = ncc_map.cols;
        int outH = ncc_map.rows;
        // Get the min and max x and y of the search window.
        int minTx = std::max(0, cx - SEARCH_RADIUS_X - templW / 2);
        int maxTx = std::min(outW - 1, cx + SEARCH_RADIUS_X - templW / 2);
        int minTy = std::max(0, cy - SEARCH_RADIUS_Y - templH / 2);
        int maxTy = std::min(outH - 1, cy + SEARCH_RADIUS_Y - templH / 2);
        // Get the dimensions of the search window.
        int searchW = maxTx - minTx + 1;
        int searchH = maxTy - minTy + 1;
        // Get the best location and value of the NCC map.
        cv::Point best_loc;
        double best_val = -1.0;
        if (searchW > 0 && searchH > 0) {
            cv::Rect search_region(minTx, minTy, searchW, searchH);
            cv::Mat ncc_roi = ncc_map(search_region);
            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(ncc_roi, &min_val, &max_val, &min_loc, &max_loc);
            best_val = max_val;
            best_loc = cv::Point(max_loc.x + minTx, max_loc.y + minTy);
        } else {
            // If the search window is invalid, use the entire NCC map.
            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(ncc_map, &min_val, &max_val, &min_loc, &max_loc);
            best_val = max_val;
            best_loc = max_loc;
        }

        // Update the current bounding box if the best NCC score is above the minimum confidence threshold.
        if (best_val >= NCC_MIN_CONFIDENCE) {
            curr_bbox.x = best_loc.x;
            curr_bbox.y = best_loc.y;
            curr_bbox.width  = templW;
            curr_bbox.height = templH;
            if (best_val >= NCC_STRONG_CONFIDENCE) {
                cv::Mat new_patch = frame_gray(curr_bbox).clone();
                cv::addWeighted(templ_gray, 1 - TEMPLATE_UPDATE_LR, new_patch, TEMPLATE_UPDATE_LR, 0.0, templ_gray);
            }
        }
        // Draw the current bounding box on the frame.
        cv::rectangle(frame, curr_bbox, cv::Scalar(0, 255, 0), 2);
        // Calculate the FPS.
        int64 curr_tick = cv::getTickCount();
        double delta_time = (curr_tick - last_tick) / cv::getTickFrequency();
        last_tick = curr_tick;
        double fps = (delta_time > 0.0) ? (1.0 / delta_time) : 0.0;
        // Count the total number of frames processed.
        total_frames++;
        // Display the FPS on the frame.
        cv::putText(frame, cv::format("FPS: %.1f", fps), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        // Display a downscaled copy sized to fit typical screens.
        cv::Mat display_frame;
        const int max_display_width = 1280;
        const int max_display_height = 720;
        double scale_w = ((double)max_display_width) / frame.cols;
        double scale_h = ((double)max_display_height) / frame.rows;
        double display_scale = std::min(1.0, std::min(scale_w, scale_h));
        cv::resize(frame, display_frame, cv::Size(), display_scale, display_scale, cv::INTER_AREA);
        cv::imshow("Tracking", display_frame);

        int key = cv::waitKey(1);
        if (key == 27) break;
    }

    // Report total time and average FPS for the interactive tracking phase.
    int64 t_end = cv::getTickCount();
    double time = double(t_end - t_start) / cv::getTickFrequency();
    double avg_fps = (time > 0.0) ? (double(total_frames) / time) : 0.0;
    std::cout << "Interactive tracking summary: "   
              << "frames=" << total_frames << ", "
              << "time=" << time << " s, "
              << "FPS=" << avg_fps << std::endl;

    return 0;
}

static int record_tracker(const std::string& video_path, std::string mode, int batch_size, const std::string& output_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return -1;
    }
    const char* display = std::getenv("DISPLAY");
    if (display == nullptr || std::string(display).empty()) {
        std::cerr << "[Headless] DISPLAY not set\n";
        return -1;
    }
    // Choose the frame to pick the template from and start tracking from.
    cv::Mat frame;
    cv::namedWindow("Frame Preview", cv::WINDOW_NORMAL);
    std::cout << "Use the preview window to pick a frame that contains the target object.\n"
              << "Press ENTER to select the current frame. Press ESC to quit.\n";

    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Reached End of Video." << std::endl;
            return -1;
        }
        cv::imshow("Frame Preview", frame);
        int key = cv::waitKey(30);

        if (key == 27) { 
            std::cout << "Template selection cancelled by user." << std::endl;
            return 0;
        }
        if (key == '\r' || key == '\n') break;
    }
    cv::destroyWindow("Frame Preview");
    // Select the template from the chosen frame
    cv::Rect roi = cv::selectROI("Select Template", frame, false, false);
    if (roi.width == 0 || roi.height == 0) {
        std::cerr << "No template selected" << std::endl;
        return -1;
    }
    cv::destroyWindow("Select Template");
    // Convert the frame to grayscale float and use the entire user-selected ROI as the template.
    cv::Mat frame_gray = to_gray(frame);
    cv::Mat templ_gray = frame_gray(roi).clone();
    // Current estimate of object location in full-frame coordinates.
    cv::Rect curr_bbox = roi;
    // Setup video writer to save annotated frames.
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) fps = 30.0;
    // Set the fourcc code for the output video.
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    if (fourcc == 0) {
        fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    }
    cv::VideoWriter writer(output_path, fourcc, fps, cv::Size(frame.cols, frame.rows));
    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video for writing: "
                  << output_path << std::endl;
        return -1;
    }
    std::vector<cv::Mat> batch_frames;
    std::vector<cv::Mat> batch_ncc;
    if (mode == "batch")
        batch_frames.reserve(batch_size);
    // Write the first frame with the selected bounding box.
    cv::rectangle(frame, curr_bbox, cv::Scalar(0, 255, 0), 2);
    writer.write(frame);
    // Start tracking from the second frame.
    int64 last_tick = cv::getTickCount();
    int64 t_start = cv::getTickCount();
    int total_frames = 1;
    std::cout << "Tracking..." << std::endl;
    while (true) {
        if (!cap.read(frame)) break;
        // Convert the frame to grayscale float.
        frame_gray = to_gray(frame);
        // Get the dimensions of the frame and template.
        int frameW = frame_gray.cols;
        int frameH = frame_gray.rows;
        int templW = templ_gray.cols;
        int templH = templ_gray.rows;
        // Compute the NCC map over the full frame.
        cv::Mat ncc_map;
        if (mode == "cpu") {
            cv::matchTemplate(frame_gray, templ_gray, ncc_map, cv::TM_CCOEFF_NORMED);
        }
        else if (mode == "shared") {
            ncc_match_shared_cuda(frame_gray, templ_gray, ncc_map);
        }
        else if (mode == "const") {
            ncc_match_const(frame_gray, templ_gray, ncc_map);
        }
        else if (mode == "const_tiled") {
            ncc_match_const_tiled(frame_gray, templ_gray, ncc_map);
        }
        else if (mode == "batch") {
            batch_frames.push_back(frame_gray.clone());
            if ((int)batch_frames.size() < batch_size) {
                cv::rectangle(frame, curr_bbox, {0,255,0}, 2);
                writer.write(frame);
                total_frames++;
                continue;
            }
            ncc_match_naive_cuda_batched(batch_frames, templ_gray, batch_ncc);
            ncc_map = batch_ncc.back();
            batch_frames.clear();
            batch_ncc.clear();
        }
        else ncc_match_naive_cuda(frame_gray, templ_gray, ncc_map);
        // Get the center of the current bounding box.
        int cx = curr_bbox.x + curr_bbox.width  / 2;
        int cy = curr_bbox.y + curr_bbox.height / 2;
        // Get the dimensions of the NCC map.
        int outW = ncc_map.cols;
        int outH = ncc_map.rows;
        // Get the min and max x and y of the search window.
        int minTx = std::max(0, cx - SEARCH_RADIUS_X - templW / 2);
        int maxTx = std::min(outW - 1, cx + SEARCH_RADIUS_X - templW / 2);
        int minTy = std::max(0, cy - SEARCH_RADIUS_Y - templH / 2);
        int maxTy = std::min(outH - 1, cy + SEARCH_RADIUS_Y - templH / 2);
        // Get the dimensions of the search window.
        int searchW = maxTx - minTx + 1;
        int searchH = maxTy - minTy + 1;
        // Get the best location and value of the NCC map.
        cv::Point best_loc;
        double best_val = -1.0;
        if (searchW > 0 && searchH > 0) {
            cv::Rect ncc_search_roi(minTx, minTy, searchW, searchH);
            cv::Mat ncc_roi = ncc_map(ncc_search_roi);
            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(ncc_roi, &min_val, &max_val, &min_loc, &max_loc);
            best_val = max_val;
            best_loc = cv::Point(max_loc.x + minTx, max_loc.y + minTy);
        } else {
            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(ncc_map, &min_val, &max_val, &min_loc, &max_loc);
            best_val = max_val;
            best_loc = max_loc;
        }
        // Update the current bounding box if the best NCC score is above the minimum confidence threshold.
        if (best_val >= NCC_MIN_CONFIDENCE) {
            curr_bbox.x = best_loc.x;
            curr_bbox.y = best_loc.y;
            curr_bbox.width  = templW;
            curr_bbox.height = templH;
            if (best_val >= NCC_STRONG_CONFIDENCE) {
                cv::Mat new_patch = frame_gray(curr_bbox).clone();
                cv::addWeighted(templ_gray, 1 - TEMPLATE_UPDATE_LR, new_patch, TEMPLATE_UPDATE_LR, 0.0, templ_gray);
            }
        }
        // Draw the current bounding box on the frame.
        cv::rectangle(frame, curr_bbox, cv::Scalar(0, 255, 0), 2);
        // Calculate the FPS.
        int64 curr_tick = cv::getTickCount();
        double delta_time = (curr_tick - last_tick) / cv::getTickFrequency();
        last_tick = curr_tick;
        double fps = (delta_time > 0.0) ? (1.0 / delta_time) : 0.0;
        // Count the total number of frames
        total_frames++;
        // Write the frame to the output video
        cv::putText(frame, cv::format("FPS: %.1f", fps), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        writer.write(frame);
    }

    // Report total time and average FPS for the interactive tracking phase.
    int64 t_end = cv::getTickCount();
    double time = double(t_end - t_start) / cv::getTickFrequency();
    double avg_fps = (time > 0.0) ? (double(total_frames) / time) : 0.0;
    std::cout << "Recorded tracking summary: "   
              << "frames=" << total_frames << ", "
              << "time=" << time << " s, "
              << "FPS=" << avg_fps << std::endl;

    return 0;
}