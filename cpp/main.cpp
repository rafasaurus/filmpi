#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>

inline void trilinearInterpolate(const unsigned char* __restrict haldData, float clutR, float clutG, float clutB, int clutSize, int clutSizeSquared, unsigned char* __restrict result) {
    int r0 = static_cast<int>(clutR);
    int g0 = static_cast<int>(clutG);
    int b0 = static_cast<int>(clutB);

    int maxIdx = clutSize - 1;
    int r1 = r0 + (r0 < maxIdx);
    int g1 = g0 + (g0 < maxIdx);
    int b1 = b0 + (b0 < maxIdx);

    float rRatio = clutR - r0;
    float gRatio = clutG - g0;
    float bRatio = clutB - b0;

    int baseG0B0 = clutSize * g0 + clutSizeSquared * b0;
    int baseG1B0 = clutSize * g1 + clutSizeSquared * b0;
    int baseG0B1 = clutSize * g0 + clutSizeSquared * b1;
    int baseG1B1 = clutSize * g1 + clutSizeSquared * b1;

    int idx000 = (r0 + baseG0B0) * 3;
    int idx100 = (r1 + baseG0B0) * 3;
    int idx010 = (r0 + baseG1B0) * 3;
    int idx110 = (r1 + baseG1B0) * 3;
    int idx001 = (r0 + baseG0B1) * 3;
    int idx101 = (r1 + baseG0B1) * 3;
    int idx011 = (r0 + baseG1B1) * 3;
    int idx111 = (r1 + baseG1B1) * 3;

    float c000_r = haldData[idx000];
    float c000_g = haldData[idx000 + 1];
    float c000_b = haldData[idx000 + 2];
    float c100_r = haldData[idx100];
    float c100_g = haldData[idx100 + 1];
    float c100_b = haldData[idx100 + 2];
    float c010_r = haldData[idx010];
    float c010_g = haldData[idx010 + 1];
    float c010_b = haldData[idx010 + 2];
    float c110_r = haldData[idx110];
    float c110_g = haldData[idx110 + 1];
    float c110_b = haldData[idx110 + 2];
    float c001_r = haldData[idx001];
    float c001_g = haldData[idx001 + 1];
    float c001_b = haldData[idx001 + 2];
    float c101_r = haldData[idx101];
    float c101_g = haldData[idx101 + 1];
    float c101_b = haldData[idx101 + 2];
    float c011_r = haldData[idx011];
    float c011_g = haldData[idx011 + 1];
    float c011_b = haldData[idx011 + 2];
    float c111_r = haldData[idx111];
    float c111_g = haldData[idx111 + 1];
    float c111_b = haldData[idx111 + 2];

    float oneMinusB = 1.0f - bRatio;
    float oneMinusG = 1.0f - gRatio;
    float oneMinusR = 1.0f - rRatio;

    float c00_r = c000_r * oneMinusB + c100_r * bRatio;
    float c00_g = c000_g * oneMinusB + c100_g * bRatio;
    float c00_b = c000_b * oneMinusB + c100_b * bRatio;
    float c01_r = c001_r * oneMinusB + c101_r * bRatio;
    float c01_g = c001_g * oneMinusB + c101_g * bRatio;
    float c01_b = c001_b * oneMinusB + c101_b * bRatio;
    float c10_r = c010_r * oneMinusB + c110_r * bRatio;
    float c10_g = c010_g * oneMinusB + c110_g * bRatio;
    float c10_b = c010_b * oneMinusB + c110_b * bRatio;
    float c11_r = c011_r * oneMinusB + c111_r * bRatio;
    float c11_g = c011_g * oneMinusB + c111_g * bRatio;
    float c11_b = c011_b * oneMinusB + c111_b * bRatio;

    float c0_r = c00_r * oneMinusG + c10_r * gRatio;
    float c0_g = c00_g * oneMinusG + c10_g * gRatio;
    float c0_b = c00_b * oneMinusG + c10_b * gRatio;
    float c1_r = c01_r * oneMinusG + c11_r * gRatio;
    float c1_g = c01_g * oneMinusG + c11_g * gRatio;
    float c1_b = c01_b * oneMinusG + c11_b * gRatio;

    result[0] = static_cast<unsigned char>(c0_r * oneMinusR + c1_r * rRatio);
    result[1] = static_cast<unsigned char>(c0_g * oneMinusR + c1_g * rRatio);
    result[2] = static_cast<unsigned char>(c0_b * oneMinusR + c1_b * rRatio);
}

void applyHaldClut(const unsigned char* __restrict haldData, int haldW, int haldH, unsigned char* __restrict imgData, int imgW, int imgH) {
    int clutSize = static_cast<int>(std::cbrt(haldW * haldH));
    int clutSizeSquared = clutSize * clutSize;
    float scale = (clutSize - 1) / 255.0f;

    constexpr int PREFETCH_DISTANCE = 16;

    for (int i = 0; i < imgH; ++i) {
        unsigned char* __restrict rowPtr = imgData + i * imgW * 3;
        for (int j = 0; j < imgW; ++j) {
            int pixelIdx = j * 3;

            if (j + PREFETCH_DISTANCE < imgW) {
                int prefetchPixelIdx = (j + PREFETCH_DISTANCE) * 3;
                float prefetchClutR = rowPtr[prefetchPixelIdx] * scale;
                float prefetchClutG = rowPtr[prefetchPixelIdx + 1] * scale;
                float prefetchClutB = rowPtr[prefetchPixelIdx + 2] * scale;

                int prefetchR0 = static_cast<int>(prefetchClutR);
                int prefetchG0 = static_cast<int>(prefetchClutG);
                int prefetchB0 = static_cast<int>(prefetchClutB);
                int prefetchMaxIdx = clutSize - 1;
                int prefetchR1 = prefetchR0 + (prefetchR0 < prefetchMaxIdx);
                int prefetchG1 = prefetchG0 + (prefetchG0 < prefetchMaxIdx);
                int prefetchB1 = prefetchB0 + (prefetchB0 < prefetchMaxIdx);

                int prefetchBaseG0B0 = clutSize * prefetchG0 + clutSizeSquared * prefetchB0;
                int prefetchBaseG1B0 = clutSize * prefetchG1 + clutSizeSquared * prefetchB0;
                int prefetchBaseG0B1 = clutSize * prefetchG0 + clutSizeSquared * prefetchB1;
                int prefetchBaseG1B1 = clutSize * prefetchG1 + clutSizeSquared * prefetchB1;

                __builtin_prefetch(&haldData[(prefetchR0 + prefetchBaseG0B0) * 3], 0, 1);
                __builtin_prefetch(&haldData[(prefetchR1 + prefetchBaseG0B0) * 3], 0, 1);
                __builtin_prefetch(&haldData[(prefetchR0 + prefetchBaseG1B0) * 3], 0, 1);
                __builtin_prefetch(&haldData[(prefetchR1 + prefetchBaseG1B0) * 3], 0, 1);
                __builtin_prefetch(&haldData[(prefetchR0 + prefetchBaseG0B1) * 3], 0, 1);
                __builtin_prefetch(&haldData[(prefetchR1 + prefetchBaseG0B1) * 3], 0, 1);
                __builtin_prefetch(&haldData[(prefetchR0 + prefetchBaseG1B1) * 3], 0, 1);
                __builtin_prefetch(&haldData[(prefetchR1 + prefetchBaseG1B1) * 3], 0, 1);
            }

            float clutR = rowPtr[pixelIdx] * scale;
            float clutG = rowPtr[pixelIdx + 1] * scale;
            float clutB = rowPtr[pixelIdx + 2] * scale;

            trilinearInterpolate(haldData, clutR, clutG, clutB, clutSize, clutSizeSquared, &rowPtr[pixelIdx]);
        }
    }
}

std::vector<std::vector<unsigned char>> extractNonExifAppSegments(const char* imagePath) {
    std::vector<std::vector<unsigned char>> appSegments;
    FILE* inputFile = fopen(imagePath, "rb");
    if (!inputFile) {
        return appSegments;
    }

    unsigned char header[2];
    if (fread(header, 1, 2, inputFile) != 2 || header[0] != 0xFF || header[1] != 0xD8) {
        fclose(inputFile);
        return appSegments;
    }

    while (!feof(inputFile)) {
        unsigned char marker[2];
        if (fread(marker, 1, 2, inputFile) != 2) break;

        if (marker[0] != 0xFF) break;

        if (marker[1] >= 0xE0 && marker[1] <= 0xEF) {
            unsigned char lenBytes[2];
            if (fread(lenBytes, 1, 2, inputFile) != 2) break;
            unsigned short segLen = (lenBytes[0] << 8) | lenBytes[1];

            if (segLen >= 2) {
                long segmentStart = ftell(inputFile) - 4;
                std::vector<unsigned char> segment(segLen + 2);
                fseek(inputFile, segmentStart, SEEK_SET);
                if (fread(segment.data(), 1, segLen + 2, inputFile) == segLen + 2) {
                    if (segLen >= 6 && memcmp(&segment[4], "Exif\0\0", 6) != 0) {
                        appSegments.push_back(std::move(segment));
                    } else if (segLen < 6) {
                        appSegments.push_back(std::move(segment));
                    }
                }
                fseek(inputFile, segmentStart + segLen + 2, SEEK_SET);
            } else {
                break;
            }
        } else if (marker[1] >= 0xC0 && marker[1] <= 0xC3) {
            break;
        } else if (marker[1] == 0xD8 || marker[1] == 0xD9) {
            break;
        } else if (marker[1] == 0xFF) {
            continue;
        } else {
            break;
        }
    }

    fclose(inputFile);
    return appSegments;
}

std::vector<unsigned char> extractExifSegment(const char* imagePath) {
    std::vector<unsigned char> exifSegment;
    FILE* inputFile = fopen(imagePath, "rb");
    if (!inputFile) {
        return exifSegment;
    }

    unsigned char header[2];
    if (fread(header, 1, 2, inputFile) != 2 || header[0] != 0xFF || header[1] != 0xD8) {
        fclose(inputFile);
        return exifSegment;
    }

    while (!feof(inputFile)) {
        unsigned char marker[2];
        if (fread(marker, 1, 2, inputFile) != 2) break;

        if (marker[0] != 0xFF) break;

        if (marker[1] == 0xE1) {
            unsigned char lenBytes[2];
            if (fread(lenBytes, 1, 2, inputFile) != 2) break;
            unsigned short segLen = (lenBytes[0] << 8) | lenBytes[1];

            if (segLen >= 6) {
                long segmentStart = ftell(inputFile) - 4;
                unsigned char identifier[6];
                if (fread(identifier, 1, 6, inputFile) == 6) {
                    if (memcmp(identifier, "Exif\0\0", 6) == 0) {
                        exifSegment.resize(segLen + 2);
                        fseek(inputFile, segmentStart, SEEK_SET);
                        if (fread(exifSegment.data(), 1, segLen + 2, inputFile) == segLen + 2) {
                            break;
                        }
                    } else {
                        fseek(inputFile, segLen - 8, SEEK_CUR);
                    }
                }
            } else {
                break;
            }
        } else if (marker[1] >= 0xC0 && marker[1] <= 0xC3) {
            break;
        } else if (marker[1] >= 0xE0 && marker[1] <= 0xEF) {
            unsigned char lenBytes[2];
            if (fread(lenBytes, 1, 2, inputFile) != 2) break;
            unsigned short segLen = (lenBytes[0] << 8) | lenBytes[1];
            if (segLen < 2) break;
            fseek(inputFile, segLen - 2, SEEK_CUR);
        }
    }

    fclose(inputFile);
    return exifSegment;
}

bool copyMetadata(const char* sourcePath, const char* destPath) {
    std::vector<unsigned char> exifSegment = extractExifSegment(sourcePath);
    if (exifSegment.empty()) {
        return true;
    }

    FILE* jpegFile = fopen(destPath, "rb");
    if (!jpegFile) {
        return false;
    }

    fseek(jpegFile, 0, SEEK_END);
    long fileSize = ftell(jpegFile);
    fseek(jpegFile, 0, SEEK_SET);

    if (fileSize < 2) {
        fclose(jpegFile);
        return false;
    }

    constexpr size_t CHUNK_SIZE = 65536;
    std::vector<unsigned char> jpegData;
    jpegData.reserve(fileSize);
    
    unsigned char buffer[CHUNK_SIZE];
    size_t totalRead = 0;
    while (totalRead < static_cast<size_t>(fileSize)) {
        size_t toRead = std::min(CHUNK_SIZE, static_cast<size_t>(fileSize) - totalRead);
        size_t bytesRead = fread(buffer, 1, toRead, jpegFile);
        if (bytesRead == 0) break;
        jpegData.insert(jpegData.end(), buffer, buffer + bytesRead);
        totalRead += bytesRead;
    }
    fclose(jpegFile);

    if (jpegData.size() < 2 || jpegData[0] != 0xFF || jpegData[1] != 0xD8) {
        return false;
    }

    size_t insertPos = 2;
    const unsigned char* __restrict dataPtr = jpegData.data();
    while (insertPos < jpegData.size() - 1) {
        if (dataPtr[insertPos] == 0xFF && dataPtr[insertPos + 1] == 0xE1) {
            insertPos += 2;
            if (insertPos + 1 < jpegData.size()) {
                unsigned short segLen = (dataPtr[insertPos] << 8) | dataPtr[insertPos + 1];
                insertPos += segLen;
            } else {
                break;
            }
        } else if (dataPtr[insertPos] == 0xFF && dataPtr[insertPos + 1] >= 0xC0 && dataPtr[insertPos + 1] <= 0xC3) {
            break;
        } else if (dataPtr[insertPos] == 0xFF && dataPtr[insertPos + 1] >= 0xE0 && dataPtr[insertPos + 1] <= 0xEF) {
            insertPos += 2;
            if (insertPos + 1 < jpegData.size()) {
                unsigned short segLen = (dataPtr[insertPos] << 8) | dataPtr[insertPos + 1];
                insertPos += segLen;
            } else {
                break;
            }
        } else if (dataPtr[insertPos] == 0xFF) {
            insertPos++;
        } else {
            insertPos++;
        }
    }

    FILE* outFile = fopen(destPath, "wb");
    if (!outFile) {
        return false;
    }

    fwrite(dataPtr, 1, insertPos, outFile);
    fwrite(exifSegment.data(), 1, exifSegment.size(), outFile);
    fwrite(dataPtr + insertPos, 1, jpegData.size() - insertPos, outFile);
    fclose(outFile);

    return true;
}

bool insertAppSegments(const char* outputPath, const std::vector<std::vector<unsigned char>>& appSegments) {
    if (appSegments.empty()) {
        return true;
    }

    FILE* jpegFile = fopen(outputPath, "rb");
    if (!jpegFile) {
        return false;
    }

    fseek(jpegFile, 0, SEEK_END);
    long fileSize = ftell(jpegFile);
    fseek(jpegFile, 0, SEEK_SET);

    if (fileSize < 2) {
        fclose(jpegFile);
        return false;
    }

    constexpr size_t CHUNK_SIZE = 65536;
    std::vector<unsigned char> jpegData;
    jpegData.reserve(fileSize);
    
    unsigned char buffer[CHUNK_SIZE];
    size_t totalRead = 0;
    while (totalRead < static_cast<size_t>(fileSize)) {
        size_t toRead = std::min(CHUNK_SIZE, static_cast<size_t>(fileSize) - totalRead);
        size_t bytesRead = fread(buffer, 1, toRead, jpegFile);
        if (bytesRead == 0) break;
        jpegData.insert(jpegData.end(), buffer, buffer + bytesRead);
        totalRead += bytesRead;
    }
    fclose(jpegFile);

    if (jpegData.size() < 2 || jpegData[0] != 0xFF || jpegData[1] != 0xD8) {
        return false;
    }

    size_t insertPos = 2;
    const unsigned char* __restrict dataPtr = jpegData.data();
    while (insertPos < jpegData.size() - 1) {
        if (dataPtr[insertPos] == 0xFF && dataPtr[insertPos + 1] >= 0xE0 && dataPtr[insertPos + 1] <= 0xEF) {
            insertPos += 2;
            if (insertPos + 1 < jpegData.size()) {
                unsigned short segLen = (dataPtr[insertPos] << 8) | dataPtr[insertPos + 1];
                insertPos += segLen;
            } else {
                break;
            }
        } else if (dataPtr[insertPos] == 0xFF && dataPtr[insertPos + 1] >= 0xC0 && dataPtr[insertPos + 1] <= 0xC3) {
            break;
        } else if (dataPtr[insertPos] == 0xFF) {
            insertPos++;
        } else {
            insertPos++;
        }
    }

    FILE* outFile = fopen(outputPath, "wb");
    if (!outFile) {
        return false;
    }

    fwrite(dataPtr, 1, insertPos, outFile);
    for (const auto& segment : appSegments) {
        fwrite(segment.data(), 1, segment.size(), outFile);
    }
    fwrite(dataPtr + insertPos, 1, jpegData.size() - insertPos, outFile);
    fclose(outFile);

    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <LUT_image_path> <photo_path> <output_path>\n";
        return 1;
    }

    const char* lutImagePath = argv[1];
    const char* imagePath = argv[2];
    const char* outputPath = argv[3];

    int haldW, haldH, haldChannels;
    unsigned char* haldData = stbi_load(lutImagePath, &haldW, &haldH, &haldChannels, 3);
    if (!haldData) {
        std::cerr << "Error: Could not open or find the HALD image at " << lutImagePath << std::endl;
        if (stbi_failure_reason()) {
            std::cerr << "Reason: " << stbi_failure_reason() << std::endl;
        }
        return 1;
    }

    int imgW, imgH, imgChannels;
    unsigned char* imgData = stbi_load(imagePath, &imgW, &imgH, &imgChannels, 3);
    if (!imgData) {
        std::cerr << "Error: Could not open or find the image at " << imagePath << std::endl;
        if (stbi_failure_reason()) {
            std::cerr << "Reason: " << stbi_failure_reason() << std::endl;
        }
        stbi_image_free(haldData);
        return 1;
    }

    applyHaldClut(haldData, haldW, haldH, imgData, imgW, imgH);

    const char* ext = strrchr(outputPath, '.');
    int success = 0;
    if (ext && (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0)) {
        success = stbi_write_jpg(outputPath, imgW, imgH, 3, imgData, 95);
        if (success) {
            std::vector<std::vector<unsigned char>> nonExifSegments = extractNonExifAppSegments(imagePath);
            copyMetadata(imagePath, outputPath);
            if (!nonExifSegments.empty()) {
                insertAppSegments(outputPath, nonExifSegments);
            }
        }
    } else {
        success = stbi_write_png(outputPath, imgW, imgH, 3, imgData, imgW * 3);
    }

    if (!success) {
        std::cerr << "Error: Failed to write output image to " << outputPath << std::endl;
        stbi_image_free(haldData);
        stbi_image_free(imgData);
        return 1;
    }

    stbi_image_free(haldData);
    stbi_image_free(imgData);
    return 0;
}
