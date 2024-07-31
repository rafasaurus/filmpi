#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <vector>

cv::Mat applyHaldClut(const cv::Mat& haldImg, const cv::Mat& img) {
    // Convert haldImg from BGR to RGB
    cv::Mat haldImgRGB;
    cv::cvtColor(haldImg, haldImgRGB, cv::COLOR_BGR2RGB);

    int haldW = haldImgRGB.cols;
    int haldH = haldImgRGB.rows;
    int channels = haldImgRGB.channels();
    int clutSize = std::round(std::pow(haldW, 1.0 / 3.0));
    float scale = (clutSize * clutSize - 1) / 255.0f;

    // Reshape haldImg into a 2D array of RGB values
    cv::Mat haldImgReshaped = haldImgRGB.reshape(1, clutSize * clutSize * clutSize);
    haldImgReshaped.convertTo(haldImgReshaped, CV_32F);

    // Convert img to float
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F);

    // Figure out the 3D CLUT indexes corresponding to the pixels in our image
    cv::Mat clutR, clutG, clutB;
    cv::Mat imgChannels[3];
    cv::split(imgFloat, imgChannels);

    clutR = imgChannels[0] * scale;
    clutG = imgChannels[1] * scale;
    clutB = imgChannels[2] * scale;

    clutR.convertTo(clutR, CV_32S);
    clutG.convertTo(clutG, CV_32S);
    clutB.convertTo(clutB, CV_32S);

    // Create the filtered image
    cv::Mat filteredImage(img.size(), CV_8UC3);
    cv::Mat filteredImageFloat = cv::Mat::zeros(img.size(), CV_32FC3);

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            int rIndex = clutR.at<int>(y, x);
            int gIndex = clutG.at<int>(y, x);
            int bIndex = clutB.at<int>(y, x);

            int index = rIndex + clutSize * clutSize * gIndex + clutSize * clutSize * clutSize * clutSize* bIndex;
            cv::Vec3f color = haldImgReshaped.at<cv::Vec3f>(index);
            filteredImageFloat.at<cv::Vec3f>(y, x) = color;
        }
    }

    filteredImageFloat.convertTo(filteredImage, CV_8UC3);
    return filteredImage;
}

int main() {
    // Load images
    cv::Mat hald = cv::imread("/home/rafael/.local/bin/luts/dehancer-fujichrome-velvia-50-k2383.png");
    cv::Mat image = cv::imread("/home/rafael/phone/DCIM/OpenCamera/IMG_20240727_115819.jpg");

    if (hald.empty() || image.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return -1;
    }

    // Apply Hald CLUT
    cv::Mat filtered = applyHaldClut(hald, image);

    // Save the result
    cv::imwrite("filtered.jpg", filtered);

    return 0;
}

