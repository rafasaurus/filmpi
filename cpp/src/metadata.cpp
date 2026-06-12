#include "filmpi/metadata.h"

#if FILMPI_WITH_EXIV2

#include <exiv2/exiv2.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

namespace filmpi {

namespace {

// stb_image_write strips everything but the pixel data, so APP2 segments
// (ICC profiles, MPF) are recovered from the source JPEG by hand; Exiv2
// does not carry them over.
std::vector<unsigned char> extractApp2Segments(const char* imagePath) {
    std::vector<unsigned char> app2Data;
    FILE* inputFile = std::fopen(imagePath, "rb");
    if (!inputFile) {
        return app2Data;
    }

    unsigned char header[2];
    if (std::fread(header, 1, 2, inputFile) != 2 || header[0] != 0xFF ||
        header[1] != 0xD8) {
        std::fclose(inputFile);
        return app2Data;
    }

    while (!std::feof(inputFile)) {
        unsigned char marker[2];
        if (std::fread(marker, 1, 2, inputFile) != 2) break;
        if (marker[0] != 0xFF) break;

        if (marker[1] == 0xE2) {
            unsigned char lenBytes[2];
            if (std::fread(lenBytes, 1, 2, inputFile) != 2) break;
            const unsigned short segLen = (lenBytes[0] << 8) | lenBytes[1];
            if (segLen < 2) break;

            const long segmentStart = std::ftell(inputFile) - 4;
            std::vector<unsigned char> segment(segLen + 2);
            std::fseek(inputFile, segmentStart, SEEK_SET);
            if (std::fread(segment.data(), 1, segLen + 2, inputFile) ==
                static_cast<size_t>(segLen) + 2) {
                app2Data.insert(app2Data.end(), segment.begin(), segment.end());
            }
            std::fseek(inputFile, segmentStart + segLen + 2, SEEK_SET);
        } else if (marker[1] >= 0xC0 && marker[1] <= 0xC3) {
            break;  // start of frame: no more APP segments
        } else if (marker[1] >= 0xE0 && marker[1] <= 0xEF) {
            unsigned char lenBytes[2];
            if (std::fread(lenBytes, 1, 2, inputFile) != 2) break;
            const unsigned short segLen = (lenBytes[0] << 8) | lenBytes[1];
            if (segLen < 2) break;
            std::fseek(inputFile, segLen - 2, SEEK_CUR);
        }
    }

    std::fclose(inputFile);
    return app2Data;
}

bool insertApp2Segments(const char* outputPath,
                        const std::vector<unsigned char>& app2Data) {
    if (app2Data.empty()) {
        return true;
    }

    FILE* jpegFile = std::fopen(outputPath, "rb");
    if (!jpegFile) {
        return false;
    }

    std::fseek(jpegFile, 0, SEEK_END);
    const long fileSize = std::ftell(jpegFile);
    std::fseek(jpegFile, 0, SEEK_SET);
    if (fileSize < 2) {
        std::fclose(jpegFile);
        return false;
    }

    std::vector<unsigned char> jpegData(fileSize);
    const size_t totalRead = std::fread(jpegData.data(), 1, fileSize, jpegFile);
    std::fclose(jpegFile);
    jpegData.resize(totalRead);

    if (jpegData.size() < 2 || jpegData[0] != 0xFF || jpegData[1] != 0xD8) {
        return false;
    }

    // Insert after the existing APP segments, before the first SOF marker.
    size_t insertPos = 2;
    const unsigned char* data = jpegData.data();
    while (insertPos + 1 < jpegData.size()) {
        if (data[insertPos] == 0xFF && data[insertPos + 1] >= 0xE0 &&
            data[insertPos + 1] <= 0xEF) {
            insertPos += 2;
            if (insertPos + 1 >= jpegData.size()) break;
            const unsigned short segLen =
                (data[insertPos] << 8) | data[insertPos + 1];
            insertPos += segLen;
        } else if (data[insertPos] == 0xFF && data[insertPos + 1] >= 0xC0 &&
                   data[insertPos + 1] <= 0xC3) {
            break;
        } else {
            ++insertPos;
        }
    }

    FILE* outFile = std::fopen(outputPath, "wb");
    if (!outFile) {
        return false;
    }
    std::fwrite(data, 1, insertPos, outFile);
    std::fwrite(app2Data.data(), 1, app2Data.size(), outFile);
    std::fwrite(data + insertPos, 1, jpegData.size() - insertPos, outFile);
    std::fclose(outFile);
    return true;
}

}  // namespace

bool metadataSupported() { return true; }

bool copyMetadata(const char* sourcePath, const char* destPath,
                  const char* lutName) {
    try {
        Exiv2::Image::UniquePtr sourceImage =
            Exiv2::ImageFactory::open(sourcePath);
        if (!sourceImage.get()) {
            return false;
        }
        sourceImage->readMetadata();

        Exiv2::Image::UniquePtr destImage = Exiv2::ImageFactory::open(destPath);
        if (!destImage.get()) {
            return false;
        }

        Exiv2::ExifData exifData = sourceImage->exifData();
        Exiv2::IptcData iptcData = sourceImage->iptcData();
        Exiv2::XmpData xmpData = sourceImage->xmpData();

        if (lutName && std::strlen(lutName) > 0) {
            exifData["Exif.Image.ImageDescription"] = lutName;
            iptcData["Iptc.Application2.Caption"] = lutName;
            xmpData["Xmp.dc.description"] = lutName;
        }

        destImage->setExifData(exifData);
        destImage->setIptcData(iptcData);
        destImage->setXmpData(xmpData);
        destImage->setComment(sourceImage->comment());
        destImage->writeMetadata();
        destImage.reset();

        const std::vector<unsigned char> app2Data =
            extractApp2Segments(sourcePath);
        if (!app2Data.empty()) {
            insertApp2Segments(destPath, app2Data);
        }
        return true;
    } catch (const Exiv2::Error&) {
        return false;
    }
}

}  // namespace filmpi

#else  // !FILMPI_WITH_EXIV2

namespace filmpi {

bool metadataSupported() { return false; }

bool copyMetadata(const char*, const char*, const char*) { return false; }

}  // namespace filmpi

#endif  // FILMPI_WITH_EXIV2
