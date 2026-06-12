#ifndef FILMPI_KERNELS_H
#define FILMPI_KERNELS_H

#include <cstddef>
#include <cstdint>

// Internal kernel interface. Each kernel applies a CLUT in place to
// `count` interleaved RGB8 pixels. `clut` points at the padded cube
// from HaldClut::data (kernels may load 4 bytes at any sample offset);
// `clutSize` is the cube edge N.
//
// All kernels compute the same result: trilinear interpolation with
// round-half-up quantization (truncate(x + 0.5)).

namespace filmpi {
namespace detail {

using KernelFn = void (*)(const std::uint8_t* clut, int clutSize,
                          std::uint8_t* pixels, std::size_t count);

void applyScalar(const std::uint8_t* clut, int clutSize,
                 std::uint8_t* pixels, std::size_t count);

#if FILMPI_HAVE_SSE41
void applySSE41(const std::uint8_t* clut, int clutSize,
                std::uint8_t* pixels, std::size_t count);
#endif

#if FILMPI_HAVE_AVX2
void applyAVX2(const std::uint8_t* clut, int clutSize,
               std::uint8_t* pixels, std::size_t count);
#endif

#if FILMPI_HAVE_NEON
void applyNEON(const std::uint8_t* clut, int clutSize,
               std::uint8_t* pixels, std::size_t count);
#endif

}  // namespace detail
}  // namespace filmpi

#endif  // FILMPI_KERNELS_H
