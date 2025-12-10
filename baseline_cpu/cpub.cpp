#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using ms    = std::chrono::duration<double, std::milli>;

// -------------------- Load video --------------------
bool loadVideo(const std::string& path, cv::VideoCapture& cap, cv::Mat& firstFrame, double& time_ms) {
    auto t0 = Clock::now();

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

    auto t1 = Clock::now();
    time_ms = ms(t1 - t0).count();
    return true;
}

// -------------------- Load cached video --------------------
bool loadCachedVideo(const std::string& path, std::vector<cv::Mat>& frames, double& time_ms) {
    auto t0 = Clock::now();

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

    auto t1 = Clock::now();
    time_ms = ms(t1 - t0).count();

    std::cout << "[Cache] Loaded frames: " << frames.size()
              << " | Time: " << time_ms << " ms\n";
    return true;
}

cv::Rect selectInitialROI(const cv::Mat& frame, double& time_ms) {
    auto t0 = Clock::now();
    cv::Rect bbox = cv::selectROI("Select Object", frame, false);
    cv::destroyWindow("Select Object");
    auto t1 = Clock::now();

    time_ms = ms(t1 - t0).count();
    return bbox;
}

cv::Ptr<cv::Tracker> createTracker(const cv::Mat& frame, const cv::Rect& bbox,  double& time_ms)
{
    auto t0 = Clock::now();
    auto tracker = cv::TrackerCSRT::create();
    tracker->init(frame, bbox);
    auto t1 = Clock::now();

    time_ms = ms(t1 - t0).count();
    return tracker;
}

bool createVideoWriter(cv::VideoWriter& out,  const cv::Mat& frame,  double fps, double& time_ms)
{
    auto t0 = Clock::now();

    out.open("../output/output_cpu_2.mp4", cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(frame.cols, frame.rows));

    auto t1 = Clock::now();
    time_ms = ms(t1 - t0).count();

    if (!out.isOpened()) {
        std::cerr << " Error: Cannot open output video file!\n";
        return false;
    }
    return true;
}

void runTracking(cv::VideoCapture& cap, cv::VideoWriter& out, cv::Ptr<cv::Tracker>& tracker, cv::Rect& bbox)
{
    cv::Mat frame;

    double t_decode = 0;
    double t_track  = 0;
    double t_draw   = 0;
    double t_write  = 0;

    int frame_count = 0;
    auto total_t0 = Clock::now();

    while (true) {
        auto t0 = Clock::now();
        cap >> frame;
        auto t1 = Clock::now();
        if (frame.empty()) break;
        t_decode += ms(t1 - t0).count();
        auto t2 = Clock::now();
        bool success = tracker->update(frame, bbox);
        auto t3 = Clock::now();
        t_track += ms(t3 - t2).count();
        auto t4 = Clock::now();
        if (success) cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);

        cv::putText(frame,
                    success ? "Tracking" : "Lost",
                    {20, 30},
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8, {0, 255, 0}, 2);
        auto t5 = Clock::now();
        t_draw += ms(t5 - t4).count();

        auto t6 = Clock::now();
        out.write(frame);
        auto t7 = Clock::now();
        t_write += ms(t7 - t6).count();

        frame_count++;
    }

    auto total_t1 = Clock::now();
    double total_ms = ms(total_t1 - total_t0).count();

    std::cout << "\n--------CPU_Baseline--------\n";
    std::cout << "Frames processed  : " << frame_count << "\n";
    std::cout << "Total time        : " << total_ms << " ms\n";
    std::cout << "Average FPS       : " << (1000.0 * frame_count / total_ms) << "\n\n";
    std::cout << "Decode time       : " << t_decode << " ms\n";
    std::cout << "Tracker time      : " << t_track  << " ms\n";
    std::cout << "Drawing time      : " << t_draw   << " ms\n";
    std::cout << "Write time        : " << t_write  << " ms\n";

    std::cout << "------------------------------\n";
}

// -------------------- MAIN --------------------
int main() {

    std::string input_video = "../data/car2.mp4";
    std::string input_cache = "../frames/car.cache";

    cv::VideoCapture cap;
    std::vector<cv::Mat> frames;
    cv::Mat firstFrame;

    bool use_cache = false;
    double t_cache = 0, t_open = 0, t_roi = 0, t_tracker = 0, t_writer = 0;

    // -------- Cache Attempt --------
    if (loadCachedVideo(input_cache, frames, t_cache)) {
        use_cache = true;
        firstFrame = frames[0];
        std::cout << " Using cached input\n";
    }
    else {
        std::cout << " Using video file\n";
        if (!loadVideo(input_video, cap, firstFrame, t_open))
            return -1;
    }

    cv::Rect bbox = selectInitialROI(firstFrame, t_roi);
    auto tracker = createTracker(firstFrame, bbox, t_tracker);
    cv::VideoWriter out;
    double fps_video = use_cache ? 30.0 : cap.get(cv::CAP_PROP_FPS);
    if (!createVideoWriter(out, firstFrame, fps_video, t_writer)) return -1;

    std::cout << "\n--init--\n";
    std::cout << "Cache load time  : " << t_cache   << " ms\n";
    std::cout << "Video open time  : " << t_open    << " ms\n";
    std::cout << "ROI select time  : " << t_roi     << " ms\n";
    std::cout << "Tracker init     : " << t_tracker << " ms\n";
    std::cout << "Writer init      : " << t_writer  << " ms\n";
    std::cout << "--------\n";

    if (!use_cache)
        runTracking(cap, out, tracker, bbox);

    out.release();
    cap.release();
    return 0;
}
