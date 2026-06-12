#include "kernels.h"

#if FILMPI_HAVE_SSE41

#include <smmintrin.h>

#include <cstring>

namespace filmpi {
namespace detail {

namespace {

// Loads one CLUT sample (RGB + 1 padding byte; the CLUT buffer is padded
// so the trailing load is safe) into a float vector {R, G, B, x}.
inline __m128 loadCorner(const std::uint8_t* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return _mm_cvtepi32_ps(
        _mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(v))));
}

inline __m128 lerp(__m128 a, __m128 b, float t) {
    return _mm_add_ps(a, _mm_mul_ps(_mm_sub_ps(b, a), _mm_set1_ps(t)));
}

}  // namespace

void applySSE41(const std::uint8_t* clut, int clutSize,
                std::uint8_t* pixels, std::size_t count) {
    const int n = clutSize;
    const int n2 = n * n;
    const int maxIdx = n - 1;
    const float scale = maxIdx / 255.0f;
    const __m128 half = _mm_set1_ps(0.5f);

    std::uint8_t* px = pixels;
    for (std::size_t i = 0; i < count; ++i, px += 3) {
        const float fr = px[0] * scale;
        const float fg = px[1] * scale;
        const float fb = px[2] * scale;

        const int r0 = static_cast<int>(fr);
        const int g0 = static_cast<int>(fg);
        const int b0 = static_cast<int>(fb);
        const int r1 = r0 + (r0 < maxIdx);
        const int g1 = g0 + (g0 < maxIdx);
        const int b1 = b0 + (b0 < maxIdx);

        const float wr = fr - r0;
        const float wg = fg - g0;
        const float wb = fb - b0;

        const int baseG0B0 = (n * g0 + n2 * b0) * 3;
        const int baseG1B0 = (n * g1 + n2 * b0) * 3;
        const int baseG0B1 = (n * g0 + n2 * b1) * 3;
        const int baseG1B1 = (n * g1 + n2 * b1) * 3;
        const int r0x3 = r0 * 3;
        const int r1x3 = r1 * 3;

        const __m128 c00 =
            lerp(loadCorner(clut + r0x3 + baseG0B0),
                 loadCorner(clut + r1x3 + baseG0B0), wr);
        const __m128 c10 =
            lerp(loadCorner(clut + r0x3 + baseG1B0),
                 loadCorner(clut + r1x3 + baseG1B0), wr);
        const __m128 c01 =
            lerp(loadCorner(clut + r0x3 + baseG0B1),
                 loadCorner(clut + r1x3 + baseG0B1), wr);
        const __m128 c11 =
            lerp(loadCorner(clut + r0x3 + baseG1B1),
                 loadCorner(clut + r1x3 + baseG1B1), wr);

        const __m128 c = lerp(lerp(c00, c10, wg), lerp(c01, c11, wg), wb);

        __m128i out = _mm_cvttps_epi32(_mm_add_ps(c, half));
        out = _mm_packus_epi16(_mm_packus_epi32(out, out), out);
        const std::uint32_t v =
            static_cast<std::uint32_t>(_mm_cvtsi128_si32(out));
        px[0] = static_cast<std::uint8_t>(v);
        px[1] = static_cast<std::uint8_t>(v >> 8);
        px[2] = static_cast<std::uint8_t>(v >> 16);
    }
}

}  // namespace detail
}  // namespace filmpi

#endif  // FILMPI_HAVE_SSE41
