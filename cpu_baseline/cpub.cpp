#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>


// Load video and read first frame
bool loadVideo(const std::string& path, cv::VideoCapture& cap, cv::Mat& firstFrame) {
    cap.open(path);
    if (!cap.isOpened()) {
        std::cerr << " Error: Cannot open video file!" << std::endl;
        return false;
    }

    cap >> firstFrame;
    if (firstFrame.empty()) {
        std::cerr << " Error: Empty first frame!" << std::endl;
        return false;
    }
    return true;
}

// Open cached video and read only the first frame.
// Subsequent frames can be streamed from the same ifstream to avoid
// keeping the entire video in memory.
bool openCachedVideo(const std::string& path,
                     std::ifstream& ifs,
                     cv::Mat& firstFrame,
                     int& width,
                     int& height,
                     int& type) {
    ifs.open(path, std::ios::binary);
    if (!ifs) return false;

    ifs.read((char*)&width, sizeof(int));
    ifs.read((char*)&height, sizeof(int));
    ifs.read((char*)&type, sizeof(int));

    if (!ifs) {
        std::cerr << " Error: Failed to read cache header!" << std::endl;
        return false;
    }

    firstFrame.create(height, width, type);
    if (!ifs.read((char*)firstFrame.data,
                  width * height * firstFrame.elemSize())) {
        std::cerr << " Error: Failed to read first cached frame!" << std::endl;
        return false;
    }

    std::cout << " Using cached input: " << path << std::endl;
    return true;
}




// Select ROI.
// On a machine with a display, this will pop up an interactive window.
// On headless machines (no DISPLAY set), fall back to a fixed ROI so the
// program can still run.
cv::Rect selectInitialROI(const cv::Mat& frame) {
    const char* display = std::getenv("DISPLAY");
    if (display == nullptr || std::string(display).empty()) {
        // Hard-coded ROI tuned for car.mp4; adjust if needed.
        int w = frame.cols;
        int h = frame.rows;
        int boxW = w / 8;
        int boxH = h / 8;
        int x = (w - boxW) / 2;
        int y = (h - boxH) / 2;
        std::cout << " [Headless] Using fixed ROI at (" << x << ", " << y
                  << ", " << boxW << ", " << boxH << ")\n";
        return cv::Rect(x, y, boxW, boxH);
    }

    cv::Rect bbox = cv::selectROI("Select Object", frame, false);
    cv::destroyWindow("Select Object");
    return bbox;
}


// Create CSRT Tracker
cv::Ptr<cv::Tracker> createTracker(const cv::Mat& frame, const cv::Rect& bbox)
{
    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();
    tracker->init(frame, bbox);
    return tracker;
}

// Create Output Video Writer
bool createVideoWriter(cv::VideoWriter& out, const cv::Mat& frame, double fps) {
    out.open("../output/output_cpu.mp4",
             cv::VideoWriter::fourcc('m','p','4','v'),
             fps,
             cv::Size(frame.cols, frame.rows));

    if (!out.isOpened()) {
        std::cerr << " Error: Cannot open output video file!" << std::endl;
        return false;
    }
    return true;
}


// Tracking Loop
void runTracking(cv::VideoCapture& cap,
                 cv::VideoWriter& out,
                 cv::Ptr<cv::Tracker>& tracker,
                 cv::Rect& bbox) {
    cv::Mat frame;
    double fps_sum = 0.0;
    int frame_count = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto start = std::chrono::high_resolution_clock::now();
        bool success = tracker->update(frame, bbox);
        auto end = std::chrono::high_resolution_clock::now();

        double frame_time = std::chrono::duration<double>(end - start).count();
        double fps = 1.0 / frame_time;

        fps_sum += fps;
        frame_count++;

        if (success) {
            cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, "Tracking",
                        {20, 30}, cv::FONT_HERSHEY_SIMPLEX,
                        0.8, {0, 255, 0}, 2);
        } else {
            cv::putText(frame, "Lost",
                        {20, 30}, cv::FONT_HERSHEY_SIMPLEX,
                        0.8, {0, 0, 255}, 2);
        }

        cv::putText(frame,
                    "FPS: " + std::to_string((int)fps),
                    {20, 60},
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8, {255, 255, 255}, 2);

        out.write(frame);
    }

    std::cout << " Average CPU FPS: "
              << (fps_sum / frame_count) << std::endl;
}


int main() {

    std::string input_video = "../data/car.mp4";
    std::string input_cache = "../frames/car.cache";

    cv::VideoCapture cap;
    std::ifstream cache_stream;
    cv::Mat firstFrame;

    bool use_cache = false;

    int width = 0, height = 0, type = 0;

    // Try cache first (streaming, not loading all frames into RAM)
    if (openCachedVideo(input_cache, cache_stream,
                        firstFrame, width, height, type)) {
        use_cache = true;
    } else {
        std::cout << " Cache not found, using video: " << input_video << std::endl;
        if (!loadVideo(input_video, cap, firstFrame)) {
            return -1;
        }
    }

    // ROI + Tracker
    cv::Rect bbox = selectInitialROI(firstFrame);
    auto tracker = createTracker(firstFrame, bbox);

    // Output video
    cv::VideoWriter out;
    double fps_video = use_cache ? 30.0 : cap.get(cv::CAP_PROP_FPS);

    if (!createVideoWriter(out, firstFrame, fps_video))
        return -1;

    double fps_sum = 0.0;
    int frame_count = 0;

    if (use_cache) {
        // Cached path: stream frames from disk to keep memory usage low.
        cv::Mat frame(height, width, type);

        while (cache_stream.read((char*)frame.data,
                                 width * height * frame.elemSize())) {
            auto start = std::chrono::high_resolution_clock::now();
            bool success = tracker->update(frame, bbox);
            auto end = std::chrono::high_resolution_clock::now();

            double fps = 1.0 / std::chrono::duration<double>(end - start).count();
            fps_sum += fps;
            frame_count++;

            if (success)
                cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);

            cv::putText(frame,
                        "FPS: " + std::to_string((int)fps),
                        {20, 60},
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.8, {255, 255, 255}, 2);

            out.write(frame);
        }

        if (frame_count > 0) {
            std::cout << " Average CPU FPS (cached): "
                      << fps_sum / frame_count << std::endl;
        }
    } else {
        runTracking(cap, out, tracker, bbox);
    }

    out.release();
    return 0;
}
