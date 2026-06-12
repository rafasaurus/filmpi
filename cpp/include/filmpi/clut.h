#ifndef FILMPI_CLUT_H
#define FILMPI_CLUT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace filmpi {

// SIMD kernel selection. Auto picks the best kernel supported by the CPU
// at runtime; the others force a specific implementation (useful for
// testing/benchmarking). Requesting a kernel that was not compiled in or
// is not supported by the CPU falls back to Scalar.
enum class Kernel {
    Auto,
    Scalar,
    SSE41,
    AVX2,
    NEON,
};

// A HALD CLUT unpacked into a lookup cube.
// data holds interleaved RGB8 samples laid out as (r + N*g + N*N*b) * 3,
// padded at the end so SIMD kernels may safely load 4 bytes at any entry.
struct HaldClut {
    std::vector<std::uint8_t> data;
    int size = 0;  // N, the cube edge length (e.g. 64 for a level-8 HALD)

    bool valid() const { return size > 1; }
};

// Builds a HaldClut from a decoded HALD image (interleaved RGB8).
// Returns an invalid HaldClut and sets *error if the dimensions do not
// form a HALD CLUT (width*height must be a perfect cube).
HaldClut makeHaldClut(const std::uint8_t* rgb, int width, int height,
                      std::string* error = nullptr);

// Applies the CLUT in place to pixelCount interleaved RGB8 pixels.
// threads == 0 uses all hardware threads; small images run single-threaded.
void applyHaldClut(const HaldClut& clut, std::uint8_t* pixels,
                   std::size_t pixelCount, int threads = 0,
                   Kernel kernel = Kernel::Auto);

// The kernel Auto would resolve to on this machine, or what `requested`
// degrades to when unavailable.
Kernel resolveKernel(Kernel requested);

const char* kernelName(Kernel kernel);

}  // namespace filmpi

#endif  // FILMPI_CLUT_H
