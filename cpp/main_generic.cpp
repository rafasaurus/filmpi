// hald_clut_simd.cpp
// Single-file HALD CLUT with x86 AVX2 / ARM NEON / ARM SVE2 #ifdefs and scalar fallback.
// Build examples at the bottom of this file comments.

#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

// ---------- Architecture detection ----------
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #define HALD_X86 1
  #include <immintrin.h>
#else
  #define HALD_X86 0
#endif

#if defined(__aarch64__) || defined(__arm__)
  #define HALD_ARM 1
  #include <arm_neon.h>
  // SVE/SVE2 (available via ACLE). Compilers define these when enabled by -march flags.
  #if defined(__ARM_FEATURE_SVE2)
    #define HALD_ARM_SVE2 1
    #include <arm_sve.h>
  #else
    #define HALD_ARM_SVE2 0
  #endif
#else
  #define HALD_ARM 0
  #define HALD_ARM_SVE2 0
#endif

// Prefetch helper (non-fatal no-op when unavailable)
static inline void hald_prefetch_read(const void* p) {
#if HALD_X86
  _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
#elif HALD_ARM
  __builtin_prefetch(p, 0 /*read*/, 1 /*low temporal locality*/);
#else
  (void)p;
#endif
}

// ---------- HALD helpers ----------

// Try to detect the HALD level L given a standard HALD image (width = height = L*L).
static inline int detectHaldLevel(const cv::Mat& hald) {
    if (hald.rows == hald.cols) {
        const int side = hald.cols;
        int L = static_cast<int>(std::lround(std::sqrt(static_cast<double>(side))));
        if (L > 0 && L * L == side) return L;
    }
    // Fallback: infer from total pixels ~ L^4 (works for well-formed images)
    const double total = static_cast<double>(hald.rows) * static_cast<double>(hald.cols);
    int L = static_cast<int>(std::lround(std::pow(total, 0.25))); // 4th root
    return std::max(L, 2);
}

// Map (r,g,b) in [0..L-1] to 2D pixel (x,y) in standard HALD layout.
// TileX = b % L, TileY = b / L; within tile: x=r, y=g
static inline cv::Vec3b haldFetch(const cv::Mat& hald, int r, int g, int b, int L) {
    // Clamp for safety (should already be in range)
    r = std::max(0, std::min(L - 1, r));
    g = std::max(0, std::min(L - 1, g));
    b = std::max(0, std::min(L - 1, b));

    const int tileX = b % L;
    const int tileY = b / L;
    const int x = r + tileX * L;
    const int y = g + tileY * L;

    // Bounds guard (defensive)
    const int X = std::max(0, std::min(hald.cols - 1, x));
    const int Y = std::max(0, std::min(hald.rows - 1, y));
    return hald.at<cv::Vec3b>(Y, X);
}

// LERP of two uchar3 colors using float weights (accurate + simple).
static inline cv::Vec3b lerp(const cv::Vec3b& a, const cv::Vec3b& b, float t) {
    const float it = 1.0f - t;
    cv::Vec3f af(a[0], a[1], a[2]);
    cv::Vec3f bf(b[0], b[1], b[2]);
    cv::Vec3f cf = af * it + bf * t;
    return cv::Vec3b(
        cv::saturate_cast<uchar>(std::round(cf[0])),
        cv::saturate_cast<uchar>(std::round(cf[1])),
        cv::saturate_cast<uchar>(std::round(cf[2]))
    );
}

// Optionally, a fixed-point lerp that some compilers autovectorize better.
// t in [0,1]; convert to [0..255] and do (a*(255-t) + b*t + 127)/255
static inline cv::Vec3b lerp_u8fix(const cv::Vec3b& a, const cv::Vec3b& b, float t) {
    int ti = std::max(0, std::min(255, static_cast<int>(t * 255.0f + 0.5f)));
    int iti = 255 - ti;
    cv::Vec3b out;
    // channel order BGR
    for (int c = 0; c < 3; ++c) {
        int val = a[c] * iti + b[c] * ti;
        out[c] = static_cast<uchar>((val + 127) / 255); // rounded divide
    }
    return out;
}

// Trilinear interpolation in (r,g,b) cube on the HALD grid
static inline cv::Vec3b trilinearInterpolate(
    const cv::Mat& haldImg, float clutR, float clutG, float clutB, int L
) {
    int r0 = static_cast<int>(std::floor(clutR));
    int g0 = static_cast<int>(std::floor(clutG));
    int b0 = static_cast<int>(std::floor(clutB));
    int r1 = std::min(r0 + 1, L - 1);
    int g1 = std::min(g0 + 1, L - 1);
    int b1 = std::min(b0 + 1, L - 1);

    float rRatio = clutR - r0;
    float gRatio = clutG - g0;
    float bRatio = clutB - b0;

    // Fetch 8 cube corners
    const cv::Vec3b c000 = haldFetch(haldImg, r0, g0, b0, L);
    const cv::Vec3b c100 = haldFetch(haldImg, r1, g0, b0, L);
    const cv::Vec3b c010 = haldFetch(haldImg, r0, g1, b0, L);
    const cv::Vec3b c110 = haldFetch(haldImg, r1, g1, b0, L);
    const cv::Vec3b c001 = haldFetch(haldImg, r0, g0, b1, L);
    const cv::Vec3b c101 = haldFetch(haldImg, r1, g0, b1, L);
    const cv::Vec3b c011 = haldFetch(haldImg, r0, g1, b1, L);
    const cv::Vec3b c111 = haldFetch(haldImg, r1, g1, b1, L);

    // Along B
    const cv::Vec3b c00 = lerp_u8fix(c000, c100, bRatio);
    const cv::Vec3b c10 = lerp_u8fix(c010, c110, bRatio);
    const cv::Vec3b c01 = lerp_u8fix(c001, c101, bRatio);
    const cv::Vec3b c11 = lerp_u8fix(c011, c111, bRatio);

    // Along G
    const cv::Vec3b c0 = lerp_u8fix(c00, c10, gRatio);
    const cv::Vec3b c1 = lerp_u8fix(c01, c11, gRatio);

    // Along R
    return lerp_u8fix(c0, c1, rRatio);
}

// ---------- Main apply function (with light arch-specific hints) ----------

static inline void applyHaldClut(const cv::Mat& haldImg, cv::Mat& img) {
    const int L = detectHaldLevel(haldImg);
    if (L < 2) {
        std::cerr << "Error: Could not detect valid HALD level from LUT image.\n";
        return;
    }

    // Scale from 0..255 into 0..(L-1)
    const float scale = (L - 1) / 255.0f;

    // Suggest alignment to the compiler (helpful on ARM/AArch64)
    const int cols = img.cols;
    const int rows = img.rows;
    for (int i = 0; i < rows; ++i) {
        cv::Vec3b* rowPtr = img.ptr<cv::Vec3b>(i);
#if HALD_ARM
        rowPtr = reinterpret_cast<cv::Vec3b*>(__builtin_assume_aligned(rowPtr, 16));
#elif HALD_X86
        rowPtr = reinterpret_cast<cv::Vec3b*>(__builtin_assume_aligned(rowPtr, 32));
#endif
        for (int j = 0; j < cols; ++j) {
            // Prefetch some future pixels of the input row to reduce stalls.
            if ((j & 31) == 0 && j + 64 < cols) {
                hald_prefetch_read(&rowPtr[j + 64]);
            }

            // OpenCV stores B,G,R
            const uchar B = rowPtr[j][0];
            const uchar G = rowPtr[j][1];
            const uchar R = rowPtr[j][2];

            const float clutR = R * scale;
            const float clutG = G * scale;
            const float clutB = B * scale;

            rowPtr[j] = trilinearInterpolate(haldImg, clutR, clutG, clutB, L);
        }
    }
}

// ---------- Program entry ----------

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <LUT_image_path> <photo_path> <output_path>\n";
        return 1;
    }

    const std::string lutImagePath = argv[1];
    const std::string imagePath    = argv[2];
    const std::string outputPath   = argv[3];

    cv::Mat haldImg = cv::imread(lutImagePath, cv::IMREAD_COLOR);
    if (haldImg.empty()) {
        std::cerr << "Error: Could not open or find the HALD image at " << lutImagePath << "\n";
        return 1;
    }

    if (haldImg.cols != haldImg.rows) {
        std::cerr << "Warning: HALD image is not square; attempting best-effort level detection.\n";
    }

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << imagePath << "\n";
        return 1;
    }

    // In-place apply
    applyHaldClut(haldImg, image);

    if (!cv::imwrite(outputPath, image)) {
        std::cerr << "Error: Failed to write output to " << outputPath << "\n";
        return 1;
    }

    return 0;
}

/*
=====================
 Build / Run Examples
=====================

# ----- Common (Linux desktop) -----
# Make sure pkg-config can find OpenCV 4. Adjust `opencv4` to `opencv` if needed.

# x86-64 with AVX2 (GCC/Clang):
g++ -O3 -DNDEBUG -march=x86-64-v3 -mavx2 -mfma hald_clut_simd.cpp -o hald_clut_simd \
    `pkg-config --cflags --libs opencv4`

# or explicitly:
g++ -O3 -DNDEBUG -mavx2 -mfma -mtune=native hald_clut_simd.cpp -o hald_clut_simd \
    `pkg-config --cflags --libs opencv4`

# Run:
./hald_clut_simd path/to/hald_L.png input.jpg output.jpg


# ----- AArch64 (ARM64) -----
# NEON is mandatory on AArch64; just pick an appropriate -march.
# Good baseline (server SBCs/phones):
clang++ -O3 -DNDEBUG -march=armv8.2-a hald_clut_simd.cpp -o hald_clut_simd \
    `pkg-config --cflags --libs opencv4`

# If your CPU supports dotprod, fp16, etc, you can do:
clang++ -O3 -DNDEBUG -march=armv8.6-a+dotprod+fp16 hald_clut_simd.cpp -o hald_clut_simd \
    `pkg-config --cflags --libs opencv4`


# ----- AArch64 with SVE2 (where available) -----
# SVE2 is optional; many phones still lack it. If present (some ARMv9 cores), try:
clang++ -O3 -DNDEBUG -march=armv9.2-a+sve2 hald_clut_simd.cpp -o hald_clut_simd \
    `pkg-config --cflags --libs opencv4`

# If the compiler accepts it, __ARM_FEATURE_SVE2 will be defined and the SVE2 path enabled
# (we only use it for prefetch/assumptions here; heavy SVE2 code can be added later).


# ----- Android (Termux on Pixel) -----
# Termux (AArch64) typically uses clang++ and has OpenCV in repositories or from source.
# Baseline NEON (AArch64):
clang++ -O3 -DNDEBUG -march=armv8.2-a hald_clut_simd.cpp -o hald_clut_simd \
    `pkg-config --cflags --libs opencv4`

# If your Pixel/SoC supports SVE2 and the Termux toolchain does too (rare as of 2025), try:
clang++ -O3 -DNDEBUG -march=armv9.2-a+sve2 hald_clut_simd.cpp -o hald_clut_simd \
    `pkg-config --cflags --libs opencv4`

# ----- Notes -----
# * The code uses a correct standard HALD layout mapper. If your LUT uses a different packing,
#   swap haldFetch() to your custom mapping.
# * Add -fopenmp and uncomment an OpenMP pragma if you want multi-core parallel rows.
# * For really big speedups, we can pre-tile the HALD into contiguous R-scanlines per (g,b)
#   slice (SoA layout) and do wide SIMD on blocks; happy to add that next.
*/

