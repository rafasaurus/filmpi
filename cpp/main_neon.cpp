#ifdef USE_NEON
#include <arm_neon.h>
#endif
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>

#ifdef USE_NEON
cv::Vec3b trilinearInterpolateNeon(const cv::Mat& haldImg, const float& clutR, const float& clutG, const float& clutB, int clutSize) {
    int r0 = std::floor(clutR);
    int r1 = std::min(r0 + 1, clutSize - 1);
    int g0 = std::floor(clutG);
    int g1 = std::min(g0 + 1, clutSize - 1);
    int b0 = std::floor(clutB);
    int b1 = std::min(b0 + 1, clutSize - 1);

    float32x4_t rRatio = vdupq_n_f32(clutR - r0);
    float32x4_t gRatio = vdupq_n_f32(clutG - g0);
    float32x4_t bRatio = vdupq_n_f32(clutB - b0);

    // Load pixel values into NEON registers
    uint8x8_t c000 = vld1_u8((uint8_t*) &haldImg.at<cv::Vec3b>(r0 + clutSize * (g0 + clutSize * b0)));
    uint8x8_t c100 = vld1_u8((uint8_t*) &haldImg.at<cv::Vec3b>(r1 + clutSize * (g0 + clutSize * b0)));
    uint8x8_t c010 = vld1_u8((uint8_t*) &haldImg.at<cv::Vec3b>(r0 + clutSize * (g1 + clutSize * b0)));
    uint8x8_t c110 = vld1_u8((uint8_t*) &haldImg.at<cv::Vec3b>(r1 + clutSize * (g1 + clutSize * b0)));
    uint8x8_t c001 = vld1_u8((uint8_t*) &haldImg.at<cv::Vec3b>(r0 + clutSize * (g0 + clutSize * b1)));
    uint8x8_t c101 = vld1_u8((uint8_t*) &haldImg.at<cv::Vec3b>(r1 + clutSize * (g0 + clutSize * b1)));
    uint8x8_t c011 = vld1_u8((uint8_t*) &haldImg.at<cv::Vec3b>(r0 + clutSize * (g1 + clutSize * b1)));
    uint8x8_t c111 = vld1_u8((uint8_t*) &haldImg.at<cv::Vec3b>(r1 + clutSize * (g1 + clutSize * b1)));

    // First interpolate along B axis
    float32x4_t c00 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c000)))), vsubq_f32(vdupq_n_f32(1.0), rRatio)),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c001)))), rRatio);
    float32x4_t c10 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c010)))), vsubq_f32(vdupq_n_f32(1.0), rRatio)),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c011)))), rRatio);
    float32x4_t c01 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c100)))), vsubq_f32(vdupq_n_f32(1.0), rRatio)),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c101)))), rRatio);
    float32x4_t c11 = vmlaq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c110)))), vsubq_f32(vdupq_n_f32(1.0), rRatio)),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(c111)))), rRatio);

    // Now interpolate along G axis
    float32x4_t c0 = vmlaq_f32(vmulq_f32(c00, vsubq_f32(vdupq_n_f32(1.0), gRatio)),
            c10, gRatio);
    float32x4_t c1 = vmlaq_f32(vmulq_f32(c01, vsubq_f32(vdupq_n_f32(1.0), gRatio)),
            c11, gRatio);

    // Finally interpolate along R axis
    float32x4_t finalColor = vmlaq_f32(vmulq_f32(c0, vsubq_f32(vdupq_n_f32(1.0), bRatio)),
            c1, bRatio);
    // Convert from float32x4_t to uint8x8_t and pack into cv::Vec3b
    uint16x4_t finalColor_u16 = vmovn_u32(vcvtq_u32_f32(finalColor));  // Convert 32-bit floats to 16-bit integers
    uint8x8_t finalColor_u8 = vqmovn_u16(vcombine_u16(finalColor_u16, finalColor_u16)); // Convert and combine to 8-bit

    cv::Vec3b finalVec3bColor; // Create an empty Vec3b to store the final color
    finalVec3bColor[0] = vget_lane_u8(finalColor_u8, 0); // Blue
    finalVec3bColor[1] = vget_lane_u8(finalColor_u8, 1); // Green
    finalVec3bColor[2] = vget_lane_u8(finalColor_u8, 2); // Red
    return finalVec3bColor;
}
#endif


// Function to apply the HALD CLUT using multithreading
void applyHaldClutPartial(const cv::Mat& haldImg, const cv::Mat& srcImg, cv::Mat& dstImg, int startRow, int endRow, int clutSize) {
    float scale = (clutSize - 1) / 255.0;
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < srcImg.cols; ++j) {
            float clutR = srcImg.at<cv::Vec3b>(i, j)[2] * scale; // BGR order
            float clutG = srcImg.at<cv::Vec3b>(i, j)[1] * scale;
            float clutB = srcImg.at<cv::Vec3b>(i, j)[0] * scale;

#ifdef USE_NEON
            dstImg.at<cv::Vec3b>(i, j) = trilinearInterpolateNeon(haldImg, clutR, clutG, clutB, clutSize);
#endif
        }
    }
}

// Main function to apply HALD CLUT with threading
void applyHaldClutThreaded(const cv::Mat& haldImg, const cv::Mat& img, cv::Mat& outputImg) {
    int haldW = haldImg.cols, haldH = haldImg.rows;
    int clutSize = std::cbrt(haldW * haldH);
    int numThreads = std::thread::hardware_concurrency();
    int rowsPerThread = img.rows / numThreads;
    std::vector<std::thread> threads;
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


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <LUT_image_path> <photo_path> <output_path>\n";
        return 1;
    }

    std::string lutImagePath = argv[1];
    std::string imagePath = argv[2];
    std::string outputPath = argv[3];

    cv::Mat haldImg = cv::imread(lutImagePath, cv::IMREAD_COLOR);
    if (haldImg.empty()) {
        std::cerr << "Error: Could not open or find the HALD image at " << lutImagePath << std::endl;
        return 1;
    }

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image at " << imagePath << std::endl;
        return 1;
    }
    cv::Mat filtered(image.size(), image.type());

    applyHaldClutThreaded(haldImg, image, filtered);

    cv::imwrite(outputPath, filtered);
    std::cout << "Application of HALD CLUT completed using parallel processing." << std::endl;
    return 0;
}
