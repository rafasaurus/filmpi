#ifdef USE_NEON
#include <arm_neon.h>
#endif
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>

#ifdef USE_NEON
// Hypothetical function to gather uchar data based on uint32x4_t indices
void gather_neon(const cv::Mat& img, uint32x4_t indices, uint8x8x3_t& result) {
    uint32_t idx[4];
    vst1q_u32(idx, indices);
    for (int i = 0; i < 4; ++i) {
        cv::Vec3b pixel = img.at<cv::Vec3b>(idx[i]);
        result.val[0][i] = pixel[0]; // B
        result.val[1][i] = pixel[1]; // G
        result.val[2][i] = pixel[2]; // R
    }
}

float32x4_t trilinearInterpolateNeonNew(const cv::Mat& haldImg, const float32x4_t& clutR, const float32x4_t& clutG, const float32x4_t& clutB, int clutSize) {
    uint32x4_t r0 = vcvtq_u32_f32(clutR);
    uint32x4_t g0 = vcvtq_u32_f32(clutG);
    uint32x4_t b0 = vcvtq_u32_f32(clutB);

    uint32x4_t r1 = vaddq_u32(r0, vdupq_n_u32(1));
    uint32x4_t g1 = vaddq_u32(g0, vdupq_n_u32(1));
    uint32x4_t b1 = vaddq_u32(b0, vdupq_n_u32(1));

    float32x4_t r1f = vcvtq_f32_u32(r1);
    float32x4_t g1f = vcvtq_f32_u32(g1);
    float32x4_t b1f = vcvtq_f32_u32(b1);

    float32x4_t ratioR = vsubq_f32(clutR, r1f);
    float32x4_t ratioG = vsubq_f32(clutG, g1f);
    float32x4_t ratioB = vsubq_f32(clutB, b1f);
    // Compute the indices for accessing the image data
    uint32x4_t index000 = vmlaq_n_u32(vmlaq_n_u32(vmulq_n_u32(b0, clutSize), g0, clutSize), r0, 1);
    uint32x4_t index100 = vmlaq_n_u32(vmlaq_n_u32(vmulq_n_u32(b0, clutSize), g0, clutSize), r1, 1);
    uint32x4_t index010 = vmlaq_n_u32(vmlaq_n_u32(vmulq_n_u32(b0, clutSize), g1, clutSize), r0, 1);
    uint32x4_t index110 = vmlaq_n_u32(vmlaq_n_u32(vmulq_n_u32(b0, clutSize), g1, clutSize), r1, 1);
    uint32x4_t index001 = vmlaq_n_u32(vmlaq_n_u32(vmulq_n_u32(b1, clutSize), g0, clutSize), r0, 1);
    uint32x4_t index101 = vmlaq_n_u32(vmlaq_n_u32(vmulq_n_u32(b1, clutSize), g0, clutSize), r1, 1);
    uint32x4_t index011 = vmlaq_n_u32(vmlaq_n_u32(vmulq_n_u32(b1, clutSize), g1, clutSize), r0, 1);
    uint32x4_t index111 = vmlaq_n_u32(vmlaq_n_u32(vmulq_n_u32(b1, clutSize), g1, clutSize), r1, 1);

    uint8x8x3_t c000, c100, c010, c110, c001, c101, c011, c111;
    gather_neon(haldImg, index000, c000);
    gather_neon(haldImg, index100, c100);
    gather_neon(haldImg, index010, c010);
    gather_neon(haldImg, index110, c110);
    gather_neon(haldImg, index001, c001);
    gather_neon(haldImg, index101, c101);
    gather_neon(haldImg, index011, c011);
    gather_neon(haldImg, index111, c111);
    
    float32x4_t result = vcvtq_f32_u32(index000);
    return result;

}

#endif

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
#else
            dstImg.at<cv::Vec3b>(i, j) = trilinearInterpolate(haldImg, clutR, clutG, clutB, clutSize);
#endif
        }
    }
}
#ifdef USE_NEON
void applyHaldClutPartialNeon(const cv::Mat& haldImg, const cv::Mat& srcImg, cv::Mat& dstImg, int startRow, int endRow, int clutSize) {
    const float scale = (clutSize - 1) / 255.0f;
    uint32x4_t clutSizeSquared = vmovq_n_u32(clutSize * clutSize);  // Create a vector with clutSize * clutSize repeated

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < srcImg.cols; j += 8) { // Process 8 pixels at a time
            // Load 8 pixels
            uint8x8x3_t src_pixel = vld3_u8(srcImg.ptr<uint8_t>(i) + j * 3);

            // Scale the indices by the factor using float conversion and multiplication
            float32x4_t clutR_low = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(src_pixel.val[2])))), scale);
            float32x4_t clutR_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(src_pixel.val[2])))), scale);
            float32x4_t clutG_low = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(src_pixel.val[1])))), scale);
            float32x4_t clutG_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(src_pixel.val[1])))), scale);
            float32x4_t clutB_low = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(src_pixel.val[0])))), scale);
            float32x4_t clutB_high = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(src_pixel.val[0])))), scale);

            // Compute indices for the lookup table
            uint32x4_t index_low = vaddq_u32(vcvtq_u32_f32(clutR_low), vmlaq_u32(vmulq_n_u32(vcvtq_u32_f32(clutG_low), clutSize), vcvtq_u32_f32(clutB_low), clutSizeSquared));
            uint32x4_t index_high = vaddq_u32(vcvtq_u32_f32(clutR_high), vmlaq_u32(vmulq_n_u32(vcvtq_u32_f32(clutG_high), clutSize), vcvtq_u32_f32(clutB_high), clutSizeSquared));

            // Load and store results
            for (int k = 0; k < 4; ++k) {
                dstImg.at<cv::Vec3b>(i, j + k) = haldImg.at<cv::Vec3b>(vgetq_lane_u32(index_low, k));
                dstImg.at<cv::Vec3b>(i, j + k + 4) = haldImg.at<cv::Vec3b>(vgetq_lane_u32(index_high, k));
            }
        }
    }
}

void applyHaldClutPartialNeon4(const cv::Mat& haldImg, const cv::Mat& srcImg, cv::Mat& dstImg, int startRow, int endRow, int clutSize) {
    const float scale = (clutSize - 1) / 255.0f;
    uint32x4_t clutSizeSquared = vmovq_n_u32(clutSize * clutSize);  // Vector with clutSize * clutSize repeated

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < srcImg.cols; j += 4) { // Process 4 pixels at a time
            // Load 4 pixels
            uint8x8x3_t src_pixel = vld3_u8(srcImg.ptr<uint8_t>(i) + j * 3);

            // Convert 8-bit integers to 16-bit and then to 32-bit floats, apply scaling
            float32x4_t clutR = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(src_pixel.val[2])))), scale);
            float32x4_t clutG = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(src_pixel.val[1])))), scale);
            float32x4_t clutB = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(src_pixel.val[0])))), scale);

            // Compute indices for the lookup table
            uint32x4_t index = vaddq_u32(vcvtq_u32_f32(clutR),
                                         vmlaq_u32(vmulq_n_u32(vcvtq_u32_f32(clutG), clutSize),
                                                   vcvtq_u32_f32(clutB), clutSizeSquared));

            // Load and store results
            for (int k = 0; k < 4; ++k) {
                dstImg.at<cv::Vec3b>(i, j + k) = haldImg.at<cv::Vec3b>(vgetq_lane_u32(index, k));
            }
        }
    }
}

void applyHaldClutPartialNeon4I(const cv::Mat& haldImg, const cv::Mat& srcImg, cv::Mat& dstImg, int startRow, int endRow, int clutSize) {
        const float scale = (clutSize - 1) / 255.0f;
    uint32x4_t clutSizeSquared = vmovq_n_u32(clutSize * clutSize);  // Vector with clutSize * clutSize repeated

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < srcImg.cols; j += 4) { // Process 4 pixels at a time
            // Load 4 pixels
            uint8x8x3_t src_pixel = vld3_u8(srcImg.ptr<uint8_t>(i) + j * 3);

            // Convert 8-bit integers to 16-bit and then to 32-bit floats, apply scaling
            float32x4_t clutR = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(src_pixel.val[2])))), scale);
            float32x4_t clutG = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(src_pixel.val[1])))), scale);
            float32x4_t clutB = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(src_pixel.val[0])))), scale);
            /*
            // Extract individual elements
            float r0 = vgetq_lane_f32(clutR, 0);
            float g0 = vgetq_lane_f32(clutG, 0);
            float b0 = vgetq_lane_f32(clutB, 0);

            float r1 = vgetq_lane_f32(clutR, 1);
            float g1 = vgetq_lane_f32(clutG, 1);
            float b1 = vgetq_lane_f32(clutB, 1);

            float r2 = vgetq_lane_f32(clutR, 2);
            float g2 = vgetq_lane_f32(clutG, 2);
            float b2 = vgetq_lane_f32(clutB, 2);

            float r3 = vgetq_lane_f32(clutR, 3);
            float g3 = vgetq_lane_f32(clutG, 3);
            float b3 = vgetq_lane_f32(clutB, 3);

            // Apply trilinear interpolation for each color
            dstImg.at<cv::Vec3b>(i, j)   = trilinearInterpolateNeon(haldImg, r0, g0, b0, clutSize);
            dstImg.at<cv::Vec3b>(i, j+1) = trilinearInterpolateNeon(haldImg, r1, g1, b1, clutSize);
            dstImg.at<cv::Vec3b>(i, j+2) = trilinearInterpolateNeon(haldImg, r2, g2, b2, clutSize);
            dstImg.at<cv::Vec3b>(i, j+3) = trilinearInterpolateNeon(haldImg, r3, g3, b3, clutSize);
            */
            float32x4_t result = trilinearInterpolateNeonNew(haldImg, clutR, clutG, clutB, clutSize);
            dstImg.at<cv::Vec3b>(i, j) = vgetq_lane_f32(result, 0);
            dstImg.at<cv::Vec3b>(i, j+1) = vgetq_lane_f32(result, 1);
            dstImg.at<cv::Vec3b>(i, j+2) = vgetq_lane_f32(result, 2);
            dstImg.at<cv::Vec3b>(i, j+3) = vgetq_lane_f32(result, 3);
        }
    }
}
#endif

// Main function to apply HALD CLUT with threading
void applyHaldClutThreaded(const cv::Mat& haldImg, const cv::Mat& img, cv::Mat& outputImg) {
    int haldW = haldImg.cols, haldH = haldImg.rows;
    int clutSize = std::cbrt(haldW * haldH);
    int numThreads = std::thread::hardware_concurrency();
    numThreads = 4;
    int rowsPerThread = img.rows / numThreads;
    std::vector<std::thread> threads;
    std::cout << "Concurrent thread count: " << numThreads << std::endl;

    for (int i = 0; i < numThreads; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i + 1) * rowsPerThread;
        if (i == numThreads - 1) {
            endRow = img.rows; // Ensure the last thread covers all remaining rows
        }
#ifdef WITH_NEON
        threads.emplace_back(applyHaldClutPartialNeon4, std::cref(haldImg), std::cref(img), std::ref(outputImg), startRow, endRow, clutSize);
#else
        threads.emplace_back(applyHaldClutPartial, std::cref(haldImg), std::cref(img), std::ref(outputImg), startRow, endRow, clutSize);
#endif
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
