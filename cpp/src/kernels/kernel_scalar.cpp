#include "kernels.h"

namespace filmpi {
namespace detail {

namespace {

inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

}  // namespace

void applyScalar(const std::uint8_t* clut, int clutSize,
                 std::uint8_t* pixels, std::size_t count) {
    const int n = clutSize;
    const int n2 = n * n;
    const int maxIdx = n - 1;
    const float scale = maxIdx / 255.0f;

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

        const std::uint8_t* c000 = clut + r0x3 + baseG0B0;
        const std::uint8_t* c100 = clut + r1x3 + baseG0B0;
        const std::uint8_t* c010 = clut + r0x3 + baseG1B0;
        const std::uint8_t* c110 = clut + r1x3 + baseG1B0;
        const std::uint8_t* c001 = clut + r0x3 + baseG0B1;
        const std::uint8_t* c101 = clut + r1x3 + baseG0B1;
        const std::uint8_t* c011 = clut + r0x3 + baseG1B1;
        const std::uint8_t* c111 = clut + r1x3 + baseG1B1;

        for (int ch = 0; ch < 3; ++ch) {
            const float c00 = lerp(c000[ch], c100[ch], wr);
            const float c10 = lerp(c010[ch], c110[ch], wr);
            const float c01 = lerp(c001[ch], c101[ch], wr);
            const float c11 = lerp(c011[ch], c111[ch], wr);
            const float c0 = lerp(c00, c10, wg);
            const float c1 = lerp(c01, c11, wg);
            px[ch] = static_cast<std::uint8_t>(lerp(c0, c1, wb) + 0.5f);
        }
    }
}

}  // namespace detail
}  // namespace filmpi
