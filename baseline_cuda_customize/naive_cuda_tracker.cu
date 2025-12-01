#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// ================= CUDA Kernels =================

__global__ void applyHannKernel(float* data, const float* hann, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    data[idx] *= hann[idx];
}

__global__ void gaussianKernel(const float* x, const float* z,
                               float* result, int width, int height,
                               float sigma_sq) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int size = width * height;
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = x[i] - z[i];
        sum_sq += diff * diff;
    }

    float norm_x = 0.0f, norm_y = 0.0f;
    for (int i = 0; i < size; i++) {
        norm_x += x[i] * x[i];
        norm_y += z[i] * z[i];
    }

    float dist = (norm_x + norm_y - 2.0f * sum_sq) / size;
    dist = fmaxf(dist, 0.0f);

    int idx = row * width + col;
    result[idx] = expf(-dist / sigma_sq);
}

// ================= KCF Tracker Class =================
class KCFTracker {
private:
    Rect roi;
    Size patch_size;
    float lambda = 0.001f;
    float sigma = 0.6f;
    float interp_factor = 0.008f;
    float output_sigma_factor = 0.10f;

    float* d_hann;
    Mat hann_window;
    Mat alphaf, yf;
    Mat z_template;

    void createHannWindow(int width, int height) {
        hann_window = Mat(height, width, CV_32F);
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                hann_window.at<float>(i, j) =
                    0.5f * (1.0f - cosf(2.0f * CV_PI * j / (width - 1))) *
                    0.5f * (1.0f - cosf(2.0f * CV_PI * i / (height - 1)));
    }

    void createGaussianPeak(int width, int height, Mat& output) {
        output = Mat(height, width, CV_32F);
        float output_sigma = sqrtf(width * height) * output_sigma_factor;
        float sigma_sq = output_sigma * output_sigma;
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
                int di = i - height / 2;
                int dj = j - width / 2;
                output.at<float>(i, j) = expf(-(di * di + dj * dj) / (2.0f * sigma_sq));
            }
    }

    void extractPatch(const Mat& gray, const Rect& roi, Mat& patch) {
        Rect safe_roi = roi & Rect(0, 0, gray.cols, gray.rows);
        patch = Mat::zeros(roi.height, roi.width, CV_32F);

        if (safe_roi.area() > 0) {
            Mat roi_patch = gray(safe_roi);
            roi_patch.convertTo(roi_patch, CV_32F, 1.0 / 255.0);
            int x_offset = safe_roi.x - roi.x;
            int y_offset = safe_roi.y - roi.y;
            roi_patch.copyTo(patch(Rect(x_offset, y_offset,
                                       safe_roi.width, safe_roi.height)));
        }

        patch = patch - 0.5f;

        float* d_patch;
        int patch_bytes = patch.total() * sizeof(float);
        cudaMalloc(&d_patch, patch_bytes);
        cudaMemcpy(d_patch, patch.data, patch_bytes, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
        applyHannKernel<<<grid, block>>>(d_patch, d_hann, roi.width, roi.height);
        cudaMemcpy(patch.data, d_patch, patch_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_patch);
    }

    Mat gaussianCorrelation(const Mat& x, const Mat& z) {
        Mat c = Mat::zeros(x.size(), CV_32F);
        Mat xf, zf;
        dft(x, xf, DFT_COMPLEX_OUTPUT);
        dft(z, zf, DFT_COMPLEX_OUTPUT);
        Mat result;
        mulSpectrums(xf, zf, result, 0, true);
        Mat c_time;
        idft(result, c_time, DFT_SCALE | DFT_REAL_OUTPUT);

        double xx = norm(x, NORM_L2SQR);
        double zz = norm(z, NORM_L2SQR);

        float sigma_sq = sigma * sigma;
        for (int i = 0; i < c_time.rows; i++)
            for (int j = 0; j < c_time.cols; j++) {
                float val = (xx + zz - 2.0f * c_time.at<float>(i, j)) / x.total();
                val = max(0.0f, val);
                c.at<float>(i, j) = expf(-val / sigma_sq);
            }
        return c;
    }

    void train(const Mat& x, bool first) {
        Mat k = gaussianCorrelation(x, x);
        Mat kf;
        dft(k, kf, DFT_COMPLEX_OUTPUT);

        Mat new_alphaf = Mat::zeros(kf.size(), kf.type());

        for (int i = 0; i < kf.rows; i++)
            for (int j = 0; j < kf.cols; j++) {
                Vec2f kf_val = kf.at<Vec2f>(i, j);
                Vec2f yf_val = yf.at<Vec2f>(i, j);

                float real = kf_val[0] + lambda;
                float imag = kf_val[1];
                float denom = real * real + imag * imag;

                if (denom > 1e-6f) {
                    new_alphaf.at<Vec2f>(i, j)[0] =
                        (yf_val[0] * real + yf_val[1] * imag) / denom;
                    new_alphaf.at<Vec2f>(i, j)[1] =
                        (yf_val[1] * real - yf_val[0] * imag) / denom;
                }
            }

        if (first) {
            alphaf = new_alphaf.clone();
            z_template = x.clone();
        } else {
            alphaf = (1.0f - interp_factor) * alphaf + interp_factor * new_alphaf;
            z_template = (1.0f - interp_factor) * z_template + interp_factor * x;
        }
    }

public:
    ~KCFTracker() { cudaFree(d_hann); }

    void init(const Mat& frame, const Rect& bbox) {
        roi = bbox;
        roi.x -= roi.width / 2;
        roi.y -= roi.height / 2;
        roi.width *= 2;
        roi.height *= 2;

        patch_size = Size(roi.width, roi.height);
        createHannWindow(patch_size.width, patch_size.height);

        int hann_bytes = hann_window.total() * sizeof(float);
        cudaMalloc(&d_hann, hann_bytes);
        cudaMemcpy(d_hann, hann_window.data, hann_bytes, cudaMemcpyHostToDevice);

        Mat y;
        createGaussianPeak(patch_size.width, patch_size.height, y);
        dft(y, yf, DFT_COMPLEX_OUTPUT);

        Mat gray, patch;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        extractPatch(gray, roi, patch);
        train(patch, true);
    }

    bool update(const Mat& frame, Rect& bbox) {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        Mat x;
        extractPatch(gray, roi, x);

        Mat k = gaussianCorrelation(x, z_template);
        Mat kf;
        dft(k, kf, DFT_COMPLEX_OUTPUT);

        Mat response_f;
        mulSpectrums(alphaf, kf, response_f, 0, false);
        Mat response;
        idft(response_f, response, DFT_SCALE | DFT_REAL_OUTPUT);

        Point maxLoc;
        double maxVal;
        minMaxLoc(response, NULL, &maxVal, NULL, &maxLoc);

        roi.x += maxLoc.x - roi.width / 2;
        roi.y += maxLoc.y - roi.height / 2;

        extractPatch(gray, roi, x);
        train(x, false);

        bbox.x = roi.x + roi.width / 4;
        bbox.y = roi.y + roi.height / 4;
        bbox.width = roi.width / 2;
        bbox.height = roi.height / 2;

        return maxVal > 0.1;
    }
};

// ================= Main + TIMING =================
int main() {
    using clk = chrono::high_resolution_clock;

    auto t_cuda0 = clk::now();
    int deviceCount; cudaGetDeviceCount(&deviceCount);
    auto t_cuda1 = clk::now();

    Mat firstFrame;
    auto t_open0 = clk::now();
    VideoCapture cap("../data/car.mp4");
    cap >> firstFrame;
    auto t_open1 = clk::now();

    auto t_roi0 = clk::now();
    Rect bbox = selectROI("Select Object", firstFrame, false);
    destroyWindow("Select Object");
    auto t_roi1 = clk::now();

    auto t_writer0 = clk::now();
    VideoWriter out("../output/output_gpu_naive.mp4",
                    VideoWriter::fourcc('m','p','4','v'),
                    cap.get(CAP_PROP_FPS),
                    firstFrame.size());
    auto t_writer1 = clk::now();

    auto t_init0 = clk::now();
    KCFTracker tracker;
    tracker.init(firstFrame, bbox);
    auto t_init1 = clk::now();

    double decode_ms = 0, update_ms = 0, draw_ms = 0, write_ms = 0;
    Mat frame;
    int frames = 0;

    while (true) {
        auto t_dec0 = clk::now();
        cap >> frame;
        auto t_dec1 = clk::now();
        if (frame.empty()) break;
        frames++;

        auto t_up0 = clk::now();
        bool ok = tracker.update(frame, bbox);
        auto t_up1 = clk::now();

        auto t_draw0 = clk::now();
        rectangle(frame, bbox, ok ? Scalar(0,0,255) : Scalar(0,255,255), 2);
        auto t_draw1 = clk::now();

        auto t_wr0 = clk::now();
        out.write(frame);
        auto t_wr1 = clk::now();

        decode_ms += chrono::duration<double, milli>(t_dec1 - t_dec0).count();
        update_ms += chrono::duration<double, milli>(t_up1  - t_up0 ).count();
        draw_ms   += chrono::duration<double, milli>(t_draw1 - t_draw0).count();
        write_ms  += chrono::duration<double, milli>(t_wr1 - t_wr0).count();
    }

    cout << "\n=========== GPU KCF PROFILING (B) ===========" << endl;
    cout << "CUDA init    : " << chrono::duration<double, milli>(t_cuda1-t_cuda0).count() << " ms\n";
    cout << "Video open   : " << chrono::duration<double, milli>(t_open1-t_open0).count() << " ms\n";
    cout << "ROI select   : " << chrono::duration<double, milli>(t_roi1-t_roi0).count() << " ms\n";
    cout << "Tracker init : " << chrono::duration<double, milli>(t_init1-t_init0).count() << " ms\n";
    cout << "Writer init  : " << chrono::duration<double, milli>(t_writer1-t_writer0).count() << " ms\n";

    cout << "\nFrames       : " << frames << endl;
    cout << "Decode avg   : " << decode_ms / frames << " ms\n";
    cout << "Update avg   : " << update_ms / frames << " ms\n";
    cout << "Draw avg     : " << draw_ms   / frames << " ms\n";
    cout << "Write avg    : " << write_ms  / frames << " ms\n";
    cout << "Avg FPS      : " << 1000.0 / (update_ms / frames) << endl;
    cout << "===========================================\n";

    return 0;
}
