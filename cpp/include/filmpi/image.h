#ifndef FILMPI_IMAGE_H
#define FILMPI_IMAGE_H

#include <cstdint>
#include <string>

namespace filmpi {

// A decoded interleaved RGB8 image. Owns its pixel buffer.
class Image {
public:
    Image() = default;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    std::uint8_t* pixels() { return data_; }
    const std::uint8_t* pixels() const { return data_; }
    int width() const { return width_; }
    int height() const { return height_; }
    std::size_t pixelCount() const {
        return static_cast<std::size_t>(width_) * height_;
    }
    bool valid() const { return data_ != nullptr; }

    // Decodes any stb-supported format to RGB8.
    static Image load(const char* path, std::string* error = nullptr);

private:
    std::uint8_t* data_ = nullptr;
    int width_ = 0;
    int height_ = 0;
};

// Encodes by extension: .jpg/.jpeg with the given quality, anything else PNG.
bool saveImage(const char* path, const Image& image, int jpegQuality = 95,
               std::string* error = nullptr);

}  // namespace filmpi

#endif  // FILMPI_IMAGE_H
