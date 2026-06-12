#include "kernels.h"

#if FILMPI_HAVE_AVX2

#include <immintrin.h>

#include <cstring>

namespace filmpi {
namespace detail {

namespace {

// Three channels of a (partially interpolated) CLUT sample for 8 pixels.
struct Channels {
    __m256 r, g, b;
};

// Gathers two CLUT corners (byte offsets idx0/idx1, 4 bytes each — the
// CLUT buffer is padded so the trailing load is safe) for 8 pixels and
// lerps them along the red axis with weight wr.
inline Channels gatherLerpR(const std::uint8_t* clut, __m256i idx0,
                            __m256i idx1, __m256 wr) {
    const int* base = reinterpret_cast<const int*>(clut);
    const __m256i v0 = _mm256_i32gather_epi32(base, idx0, 1);
    const __m256i v1 = _mm256_i32gather_epi32(base, idx1, 1);
    const __m256i ff = _mm256_set1_epi32(0xFF);

    Channels out;
    __m256 a = _mm256_cvtepi32_ps(_mm256_and_si256(v0, ff));
    __m256 b = _mm256_cvtepi32_ps(_mm256_and_si256(v1, ff));
    out.r = _mm256_add_ps(a, _mm256_mul_ps(_mm256_sub_ps(b, a), wr));

    a = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v0, 8), ff));
    b = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v1, 8), ff));
    out.g = _mm256_add_ps(a, _mm256_mul_ps(_mm256_sub_ps(b, a), wr));

    a = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v0, 16), ff));
    b = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v1, 16), ff));
    out.b = _mm256_add_ps(a, _mm256_mul_ps(_mm256_sub_ps(b, a), wr));
    return out;
}

inline Channels lerp(const Channels& a, const Channels& b, __m256 w) {
    Channels out;
    out.r = _mm256_add_ps(a.r, _mm256_mul_ps(_mm256_sub_ps(b.r, a.r), w));
    out.g = _mm256_add_ps(a.g, _mm256_mul_ps(_mm256_sub_ps(b.g, a.g), w));
    out.b = _mm256_add_ps(a.b, _mm256_mul_ps(_mm256_sub_ps(b.b, a.b), w));
    return out;
}

}  // namespace

void applyAVX2(const std::uint8_t* clut, int clutSize,
               std::uint8_t* pixels, std::size_t count) {
    const int n = clutSize;
    const int maxIdx = n - 1;
    const float scale = maxIdx / 255.0f;

    const __m256 scaleV = _mm256_set1_ps(scale);
    const __m256 halfV = _mm256_set1_ps(0.5f);
    const __m256i maxV = _mm256_set1_epi32(maxIdx);
    const __m256i nx3 = _mm256_set1_epi32(n * 3);
    const __m256i n2x3 = _mm256_set1_epi32(n * n * 3);
    const __m256i three = _mm256_set1_epi32(3);

    // Deinterleave masks: pull the R/G/B bytes of 8 packed RGB pixels
    // (24 bytes split as lo = bytes 0..15, hi = bytes 16..23) into the
    // low 8 bytes of an xmm register.
    const __m128i shufRlo =
        _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i shufRhi =
        _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i shufGlo =
        _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i shufGhi =
        _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i shufBlo =
        _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i shufBhi =
        _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    // Repack mask: drop the padding byte of four RGB0 dwords per lane.
    const __m128i pack =
        _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);

    std::uint8_t* px = pixels;
    std::size_t blocks = count / 8;
    for (std::size_t blk = 0; blk < blocks; ++blk, px += 24) {
        const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(px));
        const __m128i hi =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(px + 16));

        const __m128i r8 = _mm_or_si128(_mm_shuffle_epi8(lo, shufRlo),
                                        _mm_shuffle_epi8(hi, shufRhi));
        const __m128i g8 = _mm_or_si128(_mm_shuffle_epi8(lo, shufGlo),
                                        _mm_shuffle_epi8(hi, shufGhi));
        const __m128i b8 = _mm_or_si128(_mm_shuffle_epi8(lo, shufBlo),
                                        _mm_shuffle_epi8(hi, shufBhi));

        const __m256 fr =
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(r8)), scaleV);
        const __m256 fg =
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8)), scaleV);
        const __m256 fb =
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8)), scaleV);

        const __m256i r0 = _mm256_cvttps_epi32(fr);
        const __m256i g0 = _mm256_cvttps_epi32(fg);
        const __m256i b0 = _mm256_cvttps_epi32(fb);

        const __m256 wr = _mm256_sub_ps(fr, _mm256_cvtepi32_ps(r0));
        const __m256 wg = _mm256_sub_ps(fg, _mm256_cvtepi32_ps(g0));
        const __m256 wb = _mm256_sub_ps(fb, _mm256_cvtepi32_ps(b0));

        // cmpgt yields -1 where the cell index can step up, so subtracting
        // it increments without branching.
        const __m256i r1 = _mm256_sub_epi32(r0, _mm256_cmpgt_epi32(maxV, r0));
        const __m256i g1 = _mm256_sub_epi32(g0, _mm256_cmpgt_epi32(maxV, g0));
        const __m256i b1 = _mm256_sub_epi32(b0, _mm256_cmpgt_epi32(maxV, b0));

        // Byte offsets: (r + N*g + N*N*b) * 3.
        const __m256i r0x3 = _mm256_mullo_epi32(r0, three);
        const __m256i r1x3 = _mm256_mullo_epi32(r1, three);
        const __m256i g0n = _mm256_mullo_epi32(g0, nx3);
        const __m256i g1n = _mm256_mullo_epi32(g1, nx3);
        const __m256i b0n = _mm256_mullo_epi32(b0, n2x3);
        const __m256i b1n = _mm256_mullo_epi32(b1, n2x3);

        const __m256i baseG0B0 = _mm256_add_epi32(g0n, b0n);
        const __m256i baseG1B0 = _mm256_add_epi32(g1n, b0n);
        const __m256i baseG0B1 = _mm256_add_epi32(g0n, b1n);
        const __m256i baseG1B1 = _mm256_add_epi32(g1n, b1n);

        const Channels c00 = gatherLerpR(clut, _mm256_add_epi32(r0x3, baseG0B0),
                                         _mm256_add_epi32(r1x3, baseG0B0), wr);
        const Channels c10 = gatherLerpR(clut, _mm256_add_epi32(r0x3, baseG1B0),
                                         _mm256_add_epi32(r1x3, baseG1B0), wr);
        const Channels c01 = gatherLerpR(clut, _mm256_add_epi32(r0x3, baseG0B1),
                                         _mm256_add_epi32(r1x3, baseG0B1), wr);
        const Channels c11 = gatherLerpR(clut, _mm256_add_epi32(r0x3, baseG1B1),
                                         _mm256_add_epi32(r1x3, baseG1B1), wr);

        const Channels c0 = lerp(c00, c10, wg);
        const Channels c1 = lerp(c01, c11, wg);
        const Channels c = lerp(c0, c1, wb);

        const __m256i outR = _mm256_cvttps_epi32(_mm256_add_ps(c.r, halfV));
        const __m256i outG = _mm256_cvttps_epi32(_mm256_add_ps(c.g, halfV));
        const __m256i outB = _mm256_cvttps_epi32(_mm256_add_ps(c.b, halfV));

        // R | G<<8 | B<<16 per dword, then squeeze RGB0 down to RGB.
        const __m256i rgb0 = _mm256_or_si256(
            outR, _mm256_or_si256(_mm256_slli_epi32(outG, 8),
                                  _mm256_slli_epi32(outB, 16)));
        const __m128i s0 =
            _mm_shuffle_epi8(_mm256_castsi256_si128(rgb0), pack);
        const __m128i s1 =
            _mm_shuffle_epi8(_mm256_extracti128_si256(rgb0, 1), pack);

        // 24 bytes exactly; wider stores would clobber unprocessed input.
        std::uint32_t tail;
        _mm_storel_epi64(reinterpret_cast<__m128i*>(px), s0);
        tail = static_cast<std::uint32_t>(_mm_extract_epi32(s0, 2));
        std::memcpy(px + 8, &tail, 4);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(px + 12), s1);
        tail = static_cast<std::uint32_t>(_mm_extract_epi32(s1, 2));
        std::memcpy(px + 20, &tail, 4);
    }

    applyScalar(clut, clutSize, px, count - blocks * 8);
}

}  // namespace detail
}  // namespace filmpi

#endif  // FILMPI_HAVE_AVX2
