#include "filmpi/image.h"

#include <cstring>
#include <strings.h>
#include <utility>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace filmpi {

Image::Image(Image&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      width_(std::exchange(other.width_, 0)),
      height_(std::exchange(other.height_, 0)) {}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        stbi_image_free(data_);
        data_ = std::exchange(other.data_, nullptr);
        width_ = std::exchange(other.width_, 0);
        height_ = std::exchange(other.height_, 0);
    }
    return *this;
}

Image::~Image() { stbi_image_free(data_); }

Image Image::load(const char* path, std::string* error) {
    Image image;
    int channels = 0;
    image.data_ = stbi_load(path, &image.width_, &image.height_, &channels, 3);
    if (!image.data_ && error) {
        *error = stbi_failure_reason() ? stbi_failure_reason() : "unknown error";
    }
    return image;
}

bool saveImage(const char* path, const Image& image, int jpegQuality,
               std::string* error) {
    const char* ext = std::strrchr(path, '.');
    int ok = 0;
    if (ext && (strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0)) {
        ok = stbi_write_jpg(path, image.width(), image.height(), 3,
                            image.pixels(), jpegQuality);
    } else {
        ok = stbi_write_png(path, image.width(), image.height(), 3,
                            image.pixels(), image.width() * 3);
    }
    if (!ok && error) {
        *error = "failed to encode or write file";
    }
    return ok != 0;
}

}  // namespace filmpi
