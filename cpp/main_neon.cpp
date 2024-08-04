#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>

cv::Vec3b trilinearInterpolateNeon(const cv::Mat& haldImg, const float& clutR, const float& clutG, const float& clutB, int clutSize) {
    int r0 = std::floor(clutR);
    int r1 = std::min(r0 + 1, clutSize - 1);
    int g0 = std::floor(clutG);
    int g1 = std::min(g0 + 1, clutSize - 1);
    int b0 = std::floor(clutB);
    int b1 = std::min(b0 + 1, clutSize - 1);

    float rRatio = clutR - r0;
    float gRatio = clutG - g0;
    float bRatio = clutB - b0;

    // Load initial corner colors into NEON vectors
    uint8x8_t c000 = vld1_u8((uint8_t*)&haldImg.at<cv::Vec3b>(r0, g0 + clutSize * b0));
    uint8x8_t c100 = vld1_u8((uint8_t*)&haldImg.at<cv::Vec3b>(r1, g0 + clutSize * b0));
    uint8x8_t c010 = vld1_u8((uint8_t*)&haldImg.at<cv::Vec3b>(r0, g1 + clutSize * b0));
    uint8x8_t c110 = vld1_u8((uint8_t*)&haldImg.at<cv::Vec3b>(r1, g1 + clutSize * b0));
    uint8x8_t c001 = vld1_u8((uint8_t*)&haldImg.at<cv::Vec3b>(r0, g0 + clutSize * b1));
    uint8x8_t c101 = vld1_u8((uint8_t*)&haldImg.at<cv::Vec3b>(r1, g0 + clutSize * b1));
    uint8x8_t c011 = vld1_u8((uint8_t*)&haldImg.at<cv::Vec3b>(r0, g1 + clutSize * b1));
    uint8x8_t c111 = vld1_u8((uint8_t*)&haldImg.at<cv::Vec3b>(r1, g1 + clutSize * b1));

    // Interpolate along B (blue channel)
    float32x4_t bRatioVec = vdupq_n_f32(bRatio);
    float32x4_t oneMinusBRatio = vdupq_n_f32(1.0f - bRatio);
    float32x4_t c00 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c000)))), oneMinusBRatio),
                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c100)))), bRatioVec);
    float32x4_t c01 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c001)))), oneMinusBRatio),
                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c101)))), bRatioVec);
    float32x4_t c10 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c010)))), oneMinusBRatio),
                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c110)))), bRatioVec);
    float32x4_t c11 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c011)))), oneMinusBRatio),
                                vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c111)))), bRatioVec);

    // Interpolate along G (green channel)
    float32x4_t gRatioVec = vdupq_n_f32(gRatio);
    float32x4_t oneMinusGRatio = vdupq_n_f32(1.0f - gRatio);
    float32x4_t c0 = vmlaq_f32(vmulq_f32(c00, oneMinusGRatio), c10, gRatioVec);
    float32x4_t c1 = vmlaq_f32(vmulq_f32(c01, oneMinusGRatio), c11, gRatioVec);

    // Interpolate along R (red channel)
    float32x4_t rRatioVec = vdupq_n_f32(rRatio);
    float32x4_t oneMinusRRatio = vdupq_n_f32(1.0f - rRatio);
    float32x4_t finalColorVec = vmlaq_f32(vmulq_f32(c0, oneMinusRRatio), c1, rRatioVec);

    // Convert back to uchar and return
    uint16x4_t finalColor16 = vmovn_u32(vcvtq_u32_f32(finalColorVec));
    uint8x8_t finalColor8 = vqmovn_u16(vcombine_u16(finalColor16, finalColor16));
    cv::Vec3b finalColor;
    vst1_lane_u8(&finalColor[0], finalColor8, 0);
    vst1_lane_u8(&finalColor[1], finalColor8, 1);
    vst1_lane_u8(&finalColor[2], finalColor8, 2);
    return finalColor;
}

// Function to perform trilinear interpolation on the CLUT
cv::Vec3b trilinearInterpolate(const cv::Mat& haldImg, float clutR, float clutG, float clutB, int clutSize) {
    int r0 = std::floor(clutR);
    int r1 = std::min(r0 + 1, clutSize - 1);
    int g0 = std::floor(clutG);
    int g1 = std::min(g0 + 1, clutSize - 1);
    int b0 = std::floor(clutB);
    int b1 = std::min(b0 + 1, clutSize - 1);

    float rRatio = clutR - r0;
    float gRatio = clutG - g0;
    float bRatio = clutB - b0;

    cv::Vec3b c000 = haldImg.at<cv::Vec3b>(r0 + clutSize * (g0 + clutSize * b0));
    cv::Vec3b c100 = haldImg.at<cv::Vec3b>(r1 + clutSize * (g0 + clutSize * b0));
    cv::Vec3b c010 = haldImg.at<cv::Vec3b>(r0 + clutSize * (g1 + clutSize * b0));
    cv::Vec3b c110 = haldImg.at<cv::Vec3b>(r1 + clutSize * (g1 + clutSize * b0));
    cv::Vec3b c001 = haldImg.at<cv::Vec3b>(r0 + clutSize * (g0 + clutSize * b1));
    cv::Vec3b c101 = haldImg.at<cv::Vec3b>(r1 + clutSize * (g0 + clutSize * b1));
    cv::Vec3b c011 = haldImg.at<cv::Vec3b>(r0 + clutSize * (g1 + clutSize * b1));
    cv::Vec3b c111 = haldImg.at<cv::Vec3b>(r1 + clutSize * (g1 + clutSize * b1));

    cv::Vec3b c00 = c000 * (1 - bRatio) + c100 * bRatio;
    cv::Vec3b c01 = c001 * (1 - bRatio) + c101 * bRatio;
    cv::Vec3b c10 = c010 * (1 - bRatio) + c110 * bRatio;
    cv::Vec3b c11 = c011 * (1 - bRatio) + c111 * bRatio;

    cv::Vec3b c0 = c00 * (1 - gRatio) + c10 * gRatio;
    cv::Vec3b c1 = c01 * (1 - gRatio) + c11 * gRatio;

    cv::Vec3b finalColor = c0 * (1 - rRatio) + c1 * rRatio;
    return finalColor;
}

// Function to apply the HALD CLUT using multithreading
void applyHaldClutPartial(const cv::Mat& haldImg, const cv::Mat& srcImg, cv::Mat& dstImg, int startRow, int endRow, int clutSize) {
    float scale = (clutSize - 1) / 255.0;
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < srcImg.cols; ++j) {
            float clutR = srcImg.at<cv::Vec3b>(i, j)[2] * scale; // BGR order
            float clutG = srcImg.at<cv::Vec3b>(i, j)[1] * scale;
            float clutB = srcImg.at<cv::Vec3b>(i, j)[0] * scale;

            dstImg.at<cv::Vec3b>(i, j) = trilinearInterpolateNeon(haldImg, clutR, clutG, clutB, clutSize);
        }
    }
}

// Main function to apply HALD CLUT with threading
void applyHaldClutThreaded(const cv::Mat& haldImg, const cv::Mat& img, cv::Mat& outputImg) {
    std::cout << 0 << std::endl;
    int haldW = haldImg.cols, haldH = haldImg.rows;
    int clutSize = std::cbrt(haldW * haldH);
    int numThreads = 4;//std::thread::hardware_concurrency();
    int rowsPerThread = img.rows / numThreads;
    std::cout << 1 << std::endl;
    std::vector<std::thread> threads;
    std::cout << 2 << std::endl;
    std::cout << "Concurrent thread count: " << numThreads << std::endl;

    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i + 1) * rowsPerThread;
        if (i == numThreads - 1) {
            endRow = img.rows; // Ensure the last thread covers all remaining rows
        }
        threads.emplace_back(applyHaldClutPartial, std::cref(haldImg), std::cref(img), std::ref(outputImg), startRow, endRow, clutSize);
    }

    for (auto& thread : threads) {
        thread.join(); // Wait for all threads to complete
    }
}


int main() {
    cv::Mat haldImg = cv::imread("../dehancer-fujichrome-velvia-50-k2383.png");
    cv::Mat image = cv::imread("/home/pi/IMG_20240727_115819.jpg");
    cv::Mat filtered(image.size(), image.type());

    applyHaldClutThreaded(haldImg, image, filtered);

    cv::imwrite("filtered.jpg", filtered);
    std::cout << "Application of HALD CLUT completed using parallel processing." << std::endl;
    return 0;
}
