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

// Load cached video frames from binary file, if possible
bool loadCachedVideo(const std::string& path, std::vector<cv::Mat>& frames) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    int width, height, type;
    ifs.read((char*)&width, sizeof(int));
    ifs.read((char*)&height, sizeof(int));
    ifs.read((char*)&type, sizeof(int));

    frames.clear();

    while (true) {
        cv::Mat frame(height, width, type);
        if (!ifs.read((char*)frame.data, width * height * frame.elemSize()))
            break;
        frames.push_back(frame.clone());
    }

    int count;
    ifs.read((char*)&count, sizeof(int));

    std::cout << "[Cache] Loaded frames: " << frames.size() << std::endl;
    return true;
}




// Select ROI
cv::Rect selectInitialROI(const cv::Mat& frame) {
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
    std::vector<cv::Mat> frames;
    cv::Mat firstFrame;

    bool use_cache = false;

    // Try cache first
    if (loadCachedVideo(input_cache, frames)) {
        use_cache = true;
        firstFrame = frames[0];
        std::cout << " Using cached input: " << input_cache << std::endl;
    }
    else {
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
        // Cached path
        for (size_t i = 1; i < frames.size(); i++) {
            cv::Mat& frame = frames[i];

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

        std::cout << " Average CPU FPS (cached): "
                  << fps_sum / frame_count << std::endl;
    }
    else {
        runTracking(cap, out, tracker, bbox);
    }

    out.release();
    return 0;
}
