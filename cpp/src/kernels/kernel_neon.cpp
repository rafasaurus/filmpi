#include "kernels.h"

#if FILMPI_HAVE_NEON

#include <arm_neon.h>

#include <cstring>

namespace filmpi {
namespace detail {

namespace {

// Loads one CLUT sample (RGB + 1 padding byte; the CLUT buffer is padded
// so the trailing load is safe) into a float vector {R, G, B, x}.
inline float32x4_t loadCorner(const std::uint8_t* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    const uint8x8_t b = vreinterpret_u8_u32(vdup_n_u32(v));
    return vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(b))));
}

inline float32x4_t lerp(float32x4_t a, float32x4_t b, float t) {
    return vmlaq_n_f32(a, vsubq_f32(b, a), t);
}

}  // namespace

void applyNEON(const std::uint8_t* clut, int clutSize,
               std::uint8_t* pixels, std::size_t count) {
    const int n = clutSize;
    const int n2 = n * n;
    const int maxIdx = n - 1;
    const float scale = maxIdx / 255.0f;
    const float32x4_t half = vdupq_n_f32(0.5f);

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

        const float32x4_t c00 =
            lerp(loadCorner(clut + r0x3 + baseG0B0),
                 loadCorner(clut + r1x3 + baseG0B0), wr);
        const float32x4_t c10 =
            lerp(loadCorner(clut + r0x3 + baseG1B0),
                 loadCorner(clut + r1x3 + baseG1B0), wr);
        const float32x4_t c01 =
            lerp(loadCorner(clut + r0x3 + baseG0B1),
                 loadCorner(clut + r1x3 + baseG0B1), wr);
        const float32x4_t c11 =
            lerp(loadCorner(clut + r0x3 + baseG1B1),
                 loadCorner(clut + r1x3 + baseG1B1), wr);

        const float32x4_t c = lerp(lerp(c00, c10, wg), lerp(c01, c11, wg), wb);

        const uint32x4_t out = vcvtq_u32_f32(vaddq_f32(c, half));
        px[0] = static_cast<std::uint8_t>(vgetq_lane_u32(out, 0));
        px[1] = static_cast<std::uint8_t>(vgetq_lane_u32(out, 1));
        px[2] = static_cast<std::uint8_t>(vgetq_lane_u32(out, 2));
    }
}

}  // namespace detail
}  // namespace filmpi

#endif  // FILMPI_HAVE_NEON
