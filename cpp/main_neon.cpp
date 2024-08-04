#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>


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

            dstImg.at<cv::Vec3b>(i, j) = trilinearInterpolate(haldImg, clutR, clutG, clutB, clutSize);
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
