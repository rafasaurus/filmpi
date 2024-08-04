#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <immintrin.h>  // Include for AVX2 and earlier intrinsics

cv::Vec3b trilinearInterpolateAVX(const cv::Mat& haldImg, float clutR, float clutG, float clutB, int clutSize) {
    int r0 = std::floor(clutR);
    int r1 = std::min(r0 + 1, clutSize - 1);
    int g0 = std::floor(clutG);
    int g1 = std::min(g0 + 1, clutSize - 1);
    int b0 = std::floor(clutB);
    int b1 = std::min(b0 + 1, clutSize - 1);

    float rRatio = clutR - r0;
    float gRatio = clutG - g0;
    float bRatio = clutB - b0;

    // Create arrays of indices
    int indices[8] = {
        r0 + clutSize * (g0 + clutSize * b0),
        r1 + clutSize * (g0 + clutSize * b0),
        r0 + clutSize * (g1 + clutSize * b0),
        r1 + clutSize * (g1 + clutSize * b0),
        r0 + clutSize * (g0 + clutSize * b1),
        r1 + clutSize * (g0 + clutSize * b1),
        r0 + clutSize * (g1 + clutSize * b1),
        r1 + clutSize * (g1 + clutSize * b1)
    };

    cv::Vec3b c[8];
    for (int i = 0; i < 8; ++i) {
        c[i] = haldImg.at<cv::Vec3b>(indices[i]);
    }

    // Linear interpolation formulas using AVX2
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 c0 = _mm256_mul_ps(_mm256_set1_ps(1.0f - bRatio), _mm256_loadu_ps((float*)(&c[0])));
    __m256 c1 = _mm256_mul_ps(_mm256_set1_ps(bRatio), _mm256_loadu_ps((float*)(&c[1])));
    __m256 c00 = _mm256_add_ps(c0, c1);

    c0 = _mm256_mul_ps(_mm256_set1_ps(1.0f - gRatio), _mm256_loadu_ps((float*)(&c[2])));
    c1 = _mm256_mul_ps(_mm256_set1_ps(gRatio), _mm256_loadu_ps((float*)(&c[3])));
    __m256 c01 = _mm256_add_ps(c0, c1);

    __m256 finalColor = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(1.0f - rRatio), c00),
                                      _mm256_mul_ps(_mm256_set1_ps(rRatio), c01));

    cv::Vec3b resultColor;
    float output[8];
    _mm256_storeu_ps(output, finalColor);
    resultColor[0] = static_cast<uchar>(output[0]);
    resultColor[1] = static_cast<uchar>(output[1]);
    resultColor[2] = static_cast<uchar>(output[2]);
    return resultColor;
}

void processSlice(const cv::Mat& haldImg, cv::Mat& img, int startRow, int endRow, int clutSize) {
    float scale = (clutSize - 1) / 255.0;
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            float clutR = img.at<cv::Vec3b>(i, j)[2] * scale; // BGR order
            float clutG = img.at<cv::Vec3b>(i, j)[1] * scale;
            float clutB = img.at<cv::Vec3b>(i, j)[0] * scale;

            img.at<cv::Vec3b>(i, j) = trilinearInterpolateAVX(haldImg, clutR, clutG, clutB, clutSize);
        }
    }
}

void applyHaldClut(const cv::Mat& haldImg, cv::Mat& img) {
    int haldW = haldImg.cols, haldH = haldImg.rows;
    int clutSize = std::cbrt(haldW * haldH);

    int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);
    int rowsPerThread = img.rows / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i + 1) == numThreads ? img.rows : (i + 1) * rowsPerThread;
        threads[i] = std::thread(processSlice, std::cref(haldImg), std::ref(img), startRow, endRow, clutSize);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

int main() {
    cv::Mat haldImg = cv::imread("/home/rafael/.local/bin/luts/dehancer-fujichrome-velvia-50-k2383.png");
    cv::Mat image = cv::imread("/home/rafael/phone/DCIM/OpenCamera/IMG_20240727_115819.jpg");

    applyHaldClut(haldImg, image);

    cv::imwrite("filtered.png", image);
    std::cout << "Application of HALD CLUT completed." << std::endl;
    return 0;
}

