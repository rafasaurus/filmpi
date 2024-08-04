#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <vector>

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

void applyHaldClut(const cv::Mat& haldImg, cv::Mat& img) {
    int haldW = haldImg.cols, haldH = haldImg.rows;
    int clutSize = std::cbrt(haldW * haldH);
    float scale = (clutSize - 1) / 255.0;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            float clutR = img.at<cv::Vec3b>(i, j)[2] * scale; // BGR order
            float clutG = img.at<cv::Vec3b>(i, j)[1] * scale;
            float clutB = img.at<cv::Vec3b>(i, j)[0] * scale;

            img.at<cv::Vec3b>(i, j) = trilinearInterpolate(haldImg, clutR, clutG, clutB, clutSize);
        }
    }
}

int main() {
    cv::Mat haldImg = cv::imread("../dehancer-fujichrome-velvia-50-k2383.png", cv::IMREAD_COLOR);
    cv::Mat image = cv::imread("../orig_orig.jpg", cv::IMREAD_COLOR);

    applyHaldClut(haldImg, image);

    cv::imwrite("filtered.jpg", image);
    std::cout << "Application of HALD CLUT completed." << std::endl;
    return 0;
}
