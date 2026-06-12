#include "filmpi/clut.h"

#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include "kernels/kernels.h"

#if FILMPI_HAVE_NEON && defined(__arm__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_NEON
#define HWCAP_NEON (1 << 12)
#endif
#endif

namespace filmpi {

namespace {

// Kernels load 4 bytes per 3-byte CLUT sample; pad so the last sample
// (and a vector gather landing on it) stays in bounds.
constexpr std::size_t kClutPadding = 16;

// Below this size threading overhead outweighs the work.
constexpr std::size_t kMinPixelsPerThread = 64 * 1024;

bool cpuHasSSE41() {
#if FILMPI_HAVE_SSE41 && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("sse4.1");
#else
    return false;
#endif
}

bool cpuHasAVX2() {
#if FILMPI_HAVE_AVX2 && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

bool cpuHasNEON() {
#if FILMPI_HAVE_NEON
#if defined(__aarch64__) || defined(_M_ARM64)
    return true;  // NEON is mandatory on AArch64
#elif defined(__arm__) && defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_NEON) != 0;
#else
    return true;  // compiled for a NEON baseline
#endif
#else
    return false;
#endif
}

bool kernelAvailable(Kernel kernel) {
    switch (kernel) {
        case Kernel::Scalar: return true;
        case Kernel::SSE41:  return cpuHasSSE41();
        case Kernel::AVX2:   return cpuHasAVX2();
        case Kernel::NEON:   return cpuHasNEON();
        case Kernel::Auto:   return true;
    }
    return false;
}

detail::KernelFn kernelFunction(Kernel kernel) {
    switch (kernel) {
#if FILMPI_HAVE_SSE41
        case Kernel::SSE41: return detail::applySSE41;
#endif
#if FILMPI_HAVE_AVX2
        case Kernel::AVX2: return detail::applyAVX2;
#endif
#if FILMPI_HAVE_NEON
        case Kernel::NEON: return detail::applyNEON;
#endif
        default: return detail::applyScalar;
    }
}

}  // namespace

Kernel resolveKernel(Kernel requested) {
    if (requested == Kernel::Auto) {
        if (cpuHasAVX2()) return Kernel::AVX2;
        if (cpuHasSSE41()) return Kernel::SSE41;
        if (cpuHasNEON()) return Kernel::NEON;
        return Kernel::Scalar;
    }
    return kernelAvailable(requested) ? requested : Kernel::Scalar;
}

const char* kernelName(Kernel kernel) {
    switch (kernel) {
        case Kernel::Auto:   return "auto";
        case Kernel::Scalar: return "scalar";
        case Kernel::SSE41:  return "sse4.1";
        case Kernel::AVX2:   return "avx2";
        case Kernel::NEON:   return "neon";
    }
    return "unknown";
}

HaldClut makeHaldClut(const std::uint8_t* rgb, int width, int height,
                      std::string* error) {
    HaldClut clut;
    const long long samples = static_cast<long long>(width) * height;
    const int n = static_cast<int>(std::llround(std::cbrt(
        static_cast<double>(samples))));
    if (n < 2 || static_cast<long long>(n) * n * n != samples) {
        if (error) {
            *error = "image dimensions (" + std::to_string(width) + "x" +
                     std::to_string(height) + ") do not form a HALD CLUT";
        }
        return clut;
    }

    const std::size_t bytes = static_cast<std::size_t>(samples) * 3;
    clut.data.resize(bytes + kClutPadding, 0);
    std::memcpy(clut.data.data(), rgb, bytes);
    clut.size = n;
    return clut;
}

void applyHaldClut(const HaldClut& clut, std::uint8_t* pixels,
                   std::size_t pixelCount, int threads, Kernel kernel) {
    if (!clut.valid() || pixelCount == 0) {
        return;
    }

    const detail::KernelFn fn = kernelFunction(resolveKernel(kernel));
    const std::uint8_t* clutData = clut.data.data();
    const int n = clut.size;

    std::size_t workers = threads > 0
        ? static_cast<std::size_t>(threads)
        : std::max(1u, std::thread::hardware_concurrency());
    workers = std::min(workers,
                       std::max<std::size_t>(1, pixelCount / kMinPixelsPerThread));

    if (workers <= 1) {
        fn(clutData, n, pixels, pixelCount);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(workers);
    const std::size_t chunk = (pixelCount + workers - 1) / workers;
    for (std::size_t w = 0; w < workers; ++w) {
        const std::size_t begin = w * chunk;
        const std::size_t end = std::min(begin + chunk, pixelCount);
        if (begin >= end) break;
        pool.emplace_back(fn, clutData, n, pixels + begin * 3, end - begin);
    }
    for (std::thread& t : pool) {
        t.join();
    }
}

}  // namespace filmpi
