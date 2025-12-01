#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <iostream>

int main() {
    std::string video_path = "../data/car.mp4";
    std::string cache_path = "../frames/car.cache";

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video\n";
        return -1;
    }

    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int type   = CV_8UC3;

    std::ofstream ofs(cache_path, std::ios::binary);

    ofs.write((char*)&width,  sizeof(int));
    ofs.write((char*)&height, sizeof(int));
    ofs.write((char*)&type,   sizeof(int));

    cv::Mat frame;
    int count = 0;

    while (cap.read(frame)) {
        ofs.write((char*)frame.data, width * height * frame.elemSize());
        count++;
    }

    ofs.write((char*)&count, sizeof(int));

    std::cout << "Cached " << count << " frames.\n";
}
