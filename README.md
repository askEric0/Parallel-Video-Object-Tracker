
# Parallel Video Object Tracker 

*Eric Li*，  *Yanxin Jiang*


### Summary
We developed a high-performance visual object tracker implemented in C++ and CUDA. This system aims to achieve high performance on video object tracking, both in speed of processing and GPU utilization.

### Url
https://github.com/askEric0/Parallel-Video-Object-Tracker.git

### Folder Structure

Parallel-Video-Object-Tracker/
├─ README.md                # Project overview (you can add build/run instructions here)
├─ baseline_cpu/
│  ├─ cpub.cpp              # CPU baseline implementation
│  └─ Makefile              # Build instructions for CPU OpenCV baseline demo (Windows / nvcc)
├─ baseline_cuda/
│  ├─ cudab.cpp             # CUDA baseline implementation
│  └─ Makefile              # Build instructions for CUDA pure optical baseline demo (Windows / nvcc)
├─ data/
│  └─ ...                   # Input data, configuration files, sample datasets
├─ output/                  # Output videos
└─ tracker/
   ├─ Makefile              # Build file for tracker (Windows / nvcc)
   ├─ tracker.exp           # Experiment / run config
   ├─ include/
   │  ├─ baseline_kernel.hpp# Kernel declarations / shared headers
   │  └─ utils.hpp          # Utility helpers
   └─ src/
      ├─ baseline_kernel.cu # CUDA kernel definitions
      ├─ main.cpp           # Program entrypoint
      └─ ncc_cpu.cpp        # CPU reference implementation (NCC)


### Build & Usage

#### Prerequisites
- **CUDA Toolkit**: v12.9 (or compatible)
- **OpenCV**: 4.1.3+ (with CUDA support recommended)
- **Compiler**: MSVC (via Developer Command Prompt) or MinGW with nvcc in PATH
- **Windows**: 10 or later

#### Input Files
- **Video**: Place your `.mp4` video file at `data/car.mp4` (or modify `INPUT_VIDEO` in `src/main.cpp`)
- **Initial Bounding Box**: Selected interactively via GUI when the tracker starts

#### Output Files
- **Tracked Video**: Saved to `output/output_tracker.mp4`
- **Console Logs**: Frame count, FPS, computation time printed to stdout

#### Compile (Windows)

1. Open **Developer Command Prompt for Visual Studio** or **Git Bash** (with nvcc in PATH)
2. Navigate to the tracker folder:
   ```bash
   cd tracker
   ```
3. Build using Makefile:
   ```bash
   nmake
   ```
   Or with `make` (if using MinGW/Git Bash):
   ```bash
   make
   ```
   *Note: You will need to change Makefile to appropriate compilers and Opencv pathes if running on you own or GHC Machines*
   *The original Makefile runs originally on windows*

**Output**: `tracker.exe`

#### Run the Tracker

1. **Default mode (CUDA Naive - fastest baseline)**:
   ```bash
   tracker.exe
   ```

2. **CPU mode** (for comparison):
   ```bash
   tracker.exe --cpu
   ```

3. **CUDA Shared Memory optimization**:
   ```bash
   tracker.exe --shared
   ```

4. **CUDA Constant Memory**:
   ```bash
   tracker.exe --const
   ```

5. **CUDA Constant Memory + Tiling**:
   ```bash
   tracker.exe --const_tiled
   ```

6. **Batch processing** (process N frames at once):
   ```bash
   tracker.exe --batch=4
   ```

#### Interactive Usage

1. When the tracker starts, a window will appear showing the first frame
2. **Click and drag** to select the object bounding box (ROI)
3. **Press ENTER** or close the selection dialog to start tracking
4. The tracker will process all frames and write results to `output/output_tracker.mp4`

#### Example Session
```bash
cd tracker
nmake
tracker.exe --shared

# Output:
# --------
# NCC Tracker Starting
# Input video : ../data/car.mp4
# Mode        : shared
# --------
#
# [Select ROI in GUI window]
#
# --------
#  Tracking Complete
#  Mode       : shared
#  Frames     : 150
#  Time (sec) : 12.34
#  Computation Time (sec)  : 10.56
#  FPS        : 12.15
# --------
```

#### Configuration Parameters (edit `src/main.cpp`)
- `INPUT_VIDEO`: Path to input video (default: `../data/car.mp4`)
- `SEARCH_RADIUS_X/Y`: Search window size (default: 80×80 pixels)
- `NCC_MIN_CONFIDENCE`: Minimum match threshold (default: 0.40)
- `NCC_STRONG_CONFIDENCE`: Template update threshold (default: 0.70)
- `TEMPLATE_UPDATE_LR`: Learning rate for adaptive template (default: 0.10)
- `BATCH_SIZE`: Frames per batch in batch mode (default: 4)

#### Baseline Implementations
- **CPU baseline**: `cd baseline_cpu && make && cpub.exe`
- **CUDA baseline**: `cd baseline_cuda && make && cudab.exe`

#### Performance Notes
- **--shared**: Uses shared memory tiling for better memory coalescing
- **--const_tiled**: Combines constant memory + shared memory (fastest for small templates)
- **--batch=N**: Processes N frames concurrently; higher N = better GPU utilization but more memory

#### Troubleshooting
- **"Cannot open video"**: Ensure `data/car.mp4` exists, or modify `INPUT_VIDEO`
- **"Cannot open video writer"**: Ensure `output/` directory exists
- **CUDA compilation error**: Check CUDA toolkit installation and `nvcc` in PATH
- **OpenCV linking error**: Verify library paths in `Makefile` match your OpenCV installation




### Background
Object tracking is a computer vision technique that identifies and follows a specific object across consecutive frames in a video. A tracker typically begins with an initialized bounding box in the first frame, extracts a representation of the object, and then searches for the best matching region in each subsequent frame.

The search and evaluation process for object tracking is computationally intensive. For each frame, the tracker may need to evaluate hundreds of candidate regions. As a result, even relatively simple tracking algorithms can quickly exceed the processing capacity of a CPU when real-time frame rates are required. Motion and noise further increase the difficulty and may require broader search regions, adding more computation per frame.

Most of these computations can be parallelized. Evaluating one candidate regions doesn't depend on others. This independence enables massive spatial parallelism, allowing thousands of threads to process different parts of the search area simultaneously. Since similarity computation dominates the overall runtime, accelerating this component through parallelism provides the greatest performance benefit. Our current idea is shown as below:

### The Challenge
Object tracking is challenging to parallelize because the overall task contains both parallel and sequential components. While the visual similarity computation between the template and the incoming video frames is  parallel, the high-level tracking logic is inherently sequential: the bounding box in frame t+1 depends on the position found in frame t. This indicates that the project must separate parallelizable workload from the sequential state update, and hybrid the parallel and sequential desgin.

The workload also places heavy pressure on memory bandwidth. Template matching requires sliding a window across each frame, causing significant global memory access with overlapping regions and low arithmetic intensity. Without careful GPU optimization, such as shared memory tiling, coalesced memory access, and block-level synchronization, the kernel becomes memory-bound and performs poorly. These memory characteristics—large 2D arrays, overlapping reads, and strided access patterns—make this a difficult problem to efficiently map onto CUDA’s hierarchical memory architecture. Divergent execution at image borders and the challenge of fitting shared tiles into limited SM memory further increase the complexity of the kernel design.

### Resources
We develop and run the project primarily on a Windows and a Linux machine equipped with an NVIDIA GPU. CUDA programming requires access to NVIDIA hardware, and these GPUs provide more than enough computational capability to run and evaluate the performance of my kernels.

The ideal choice of machine is the GHC cluster machines for benchmarking and performance testing.

#### Software
CUDA 11.7 and recent NVIDIA driver support will be required. The machine should be able to run C++ and CUDA, that is, with C++ and nvcc compiler installed.

#### Code
The project will be mostly starting and take advantage of OpenCV packages for frame loading and basic image manipulation. No additional external object tracking libraries (e.g., OpenCV Tracker API, DeepSORT, OpenCV DNN) will be used because they would hide the parallel compute parts of the project, and provides no help in batching frames.

#### Potential Future Needs
The potential needs include better and more recent high performance GPUs, and may need to increase the quota of AFS storage spaces, if we plan to consitently use GHC machines and AFS.

### Goals and Deliverables
The project goal is to build a GPU paralleled tracking system. It will be a CUDA accelerated single object tracker, that is designed to take advantage of GPU parallelism, memory optimization, and it will improve over CPU baseline and non-optimized GPU baseline. The tracker follows a target object in video across a video sequence by comparing a reference template against incoming video frames and locating the best matching region.

To take advantage of GPU parallelism, the tracker system processes batches of video frames simultaneously on GPU. While object tracking requires sequential updates over stream of frames, the visual similarity comparison if fully independent (between template and frames). Batching frames will maximize GPU utilization and increase throughput.

We plan to improve the speed of video processing with at least 5$\times$ from a baseline when tracking an element, and will show a demo video during the day of presentation.

#### Milestone Goals (50\%, 75\%, 100\%, 125\%, 150\%)

#### 50% Goal: Baseline and Infrastructure Completed
- Implement a CPU-based tracker for the reference of correctness.
- Implement a naive CUDA kernel for similarity computation without optimization
\end{itemize}

#### 75% Goal: Initial GPU Speedup
- Add batched frame processing to improve GPU utilization.
- Implement shared memory tiling ~~and block-level synchronization~~ for CUDA optimization.

#### 100% Goal (Target Completion): Optimized CUDA Tracker
- ~~Achieve about $5\times$ speedup over CPU baseline.~~
- Achieve about $1.2\times-1.5\times$ speedup over CPU baseline
- Demonstrate stable real-time or near-real-time tracking on short videos.

#### 125% Goal: Extended Optimizations and Robustness
- Real-time tracking by tuning batch size and accept streaming.
- Achieve more than $5\times$ speedup on parallel implementation.

#### 150% Goal: Advanced Features Improvements
- Achieve multiple-object tracking by running several trackers concurrently.

### Platform Choice
CUDA on an NVIDIA GPU is an ideal platform for this project because the core workload—template matching and pixel-level similarity—maps naturally to thousands of parallel threads. CUDA gives the control, performance, and parallel capabilities necessary to build an efficient and scalable tracking system.

### Schedule

The schedule will be roughly divided into 4 weeks' workload. Week 4 is short hence allocated for project report writing.

Updated Schedule:
#### Dec 1 -- Dec 4:
- Implement batched frame processing to improve GPU utilization. (Eric + Yanxin)
- Try to apply lightweight kernel optimizations, such as basic shared-memory tiling, if it indeed helps optimization. (Eric + Yanxin)
- Try to achieve a modest speedup of 1.2x-1.5x over the current CUDA version. (Eric \& Yanxin)

#### Dec 5 -- Dec 8:
- Clean up and finalize the code for the final report. (Eric + Yanxin)
- Demonstrate stable real-time or near-real-time tracking on representative video examples. (Eric + Yanxin)

Original Schedule:
#### Week 1:
- Implement CPU baseline tracker (template matching / NCC) for correctness reference. (Yanxin + Eric)
- Set up project structure, build system (CMake), and OpenCV I/O + visualization. (Eric)
- Write a naïve CUDA kernel for template matching (no optimization). (Yanxin + Eric)
- Verify correctness (Yanxin)

#### Week 2:
- Add batched frame processing (process 4–8 frames per kernel launch). (Yanxin)
- Integrate GPU similarity output back into sequential tracking update. (Yanxin)
- Implement shared memory tiling + block-level synchronization for CUDA optimization. (Eric)

#### Week 3:
- Finalize CUDA optimizations (coalesced reads, minimizing redundant loads). (Yanxin + Eric)
- Ensure stable real-time or near-real-time tracking on short videos. (Yanxin + Eric)
- Test on videos, verify correctness. (Eric)

#### Week 4:
- Project Report Writing. (Yanxin + Eric)
- Optional: Real-time tracking by tuning batch size and accept streaming. (Yanxin + Eric)




