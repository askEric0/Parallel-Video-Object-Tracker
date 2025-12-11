#include <opencv2/opencv.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

Rect selectROIInit(const Mat& frame) {
    Rect bbox = selectROI("Select Object", frame, false);
    destroyWindow("Select Object");
    return bbox;
}
Rect clampBBox(Rect b, int W, int H) {
    b.x = max(0, min(b.x, W - b.width));
    b.y = max(0, min(b.y, H - b.height));
    return b;
}
float median(vector<float>& v) {
    if (v.empty()) return 0;
    nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    return v[v.size()/2];
}
int main() {

    string input = "../data/car.mp4";
    string output = "../output/output_cuda_baseline.mp4";

    VideoCapture cap(input);
    if (!cap.isOpened()) {
        cout << " Cannot open video\n";
        return -1;
    }

    Mat first;
    cap >> first;
    if (first.empty()) return -1;

    int W = first.cols;
    int H = first.rows;

    Rect bbox = selectROIInit(first);
    if (bbox.width <= 0 || bbox.height <= 0) return -1;

    VideoWriter out(output,VideoWriter::fourcc('m','p','4','v'), cap.get(CAP_PROP_FPS),Size(W, H));

    Ptr<cuda::FarnebackOpticalFlow> flow = cuda::FarnebackOpticalFlow::create();
    cuda::GpuMat d_prev, d_curr, d_flow;
    cuda::GpuMat d_fx, d_fy;

    Mat prev_gray, curr_gray;
    cvtColor(first, prev_gray, COLOR_BGR2GRAY);
    d_prev.upload(prev_gray);

    int frame_count = 0;
    auto t0 = chrono::high_resolution_clock::now();
    double tot = 0.0;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, curr_gray, COLOR_BGR2GRAY);
        d_curr.upload(curr_gray);

        auto t4 = chrono::high_resolution_clock::now();

        flow->calc(d_prev, d_curr, d_flow);

        cuda::GpuMat parts[2];
        cuda::split(d_flow, parts);
        d_fx = parts[0];
        d_fy = parts[1];
        Mat fx_cpu, fy_cpu;
        d_fx(bbox).download(fx_cpu);
        d_fy(bbox).download(fy_cpu);

        vector<float> dxs, dys;
        for (int y = 0; y < fy_cpu.rows; y++) {
            for (int x = 0; x < fx_cpu.cols; x++) {
                float dx = fx_cpu.at<float>(y, x);
                float dy = fy_cpu.at<float>(y, x);
                float mag = sqrt(dx*dx + dy*dy);

                if (mag > 0.5f && mag < 25.0f) {
                    dxs.push_back(dx);
                    dys.push_back(dy);
                }
            }
        }
        if (dxs.size() > 0.15 * bbox.area()) {
            float dx = median(dxs);
            float dy = median(dys);

            bbox.x += int(dx);
            bbox.y += int(dy);
            bbox = clampBBox(bbox, W, H);
        }



        rectangle(frame, bbox, Scalar(0,0,255), 2);
        putText(frame, "PURE CUDA FLOW", {20,30}, FONT_HERSHEY_SIMPLEX, 0.8, {0,255,0}, 2);

        auto t3 = chrono::high_resolution_clock::now();
        tot += chrono::duration<double, milli>(t3 - t4).count();

        out.write(frame);
        d_curr.copyTo(d_prev);
        frame_count++;
    }

    auto t1 = chrono::high_resolution_clock::now();
    double total_ms = chrono::duration<double, milli>(t1 - t0).count();

    cout << "\n Pure Cuda optical Flow baseline\n";
    cout << "Frames  : " << frame_count << endl;
    cout << "Time ms : " << total_ms << endl;
    cout << "Time computation s : " << tot << endl;
    cout << "FPS     : " << (1000.0 * frame_count / total_ms) << endl;

    return 0;
}
