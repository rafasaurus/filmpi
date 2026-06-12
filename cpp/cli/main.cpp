#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "filmpi/clut.h"
#include "filmpi/image.h"
#include "filmpi/metadata.h"

#ifndef FILMPI_VERSION
#define FILMPI_VERSION "dev"
#endif

namespace {

void printUsage(const char* prog) {
    std::printf(
        "Usage: %s [options] <hald_clut> <input_image> <output_image>\n"
        "\n"
        "Applies a HALD CLUT to an image. Output format follows the output\n"
        "extension: .jpg/.jpeg writes JPEG (with metadata copied from the\n"
        "input when supported), anything else writes PNG.\n"
        "\n"
        "Options:\n"
        "  -q, --quality <1-100>  JPEG quality (default 95)\n"
        "  -t, --threads <n>      worker threads (default: all cores)\n"
        "  -k, --kernel <name>    auto|scalar|sse41|avx2|neon (default auto)\n"
        "  -M, --no-metadata      do not copy EXIF/IPTC/XMP metadata\n"
        "  -v, --verbose          print kernel and timing details\n"
        "      --version          print version and exit\n"
        "  -h, --help             print this help and exit\n",
        prog);
}

bool parseKernel(const char* name, filmpi::Kernel* out) {
    static const struct { const char* name; filmpi::Kernel kernel; } table[] = {
        {"auto", filmpi::Kernel::Auto},   {"scalar", filmpi::Kernel::Scalar},
        {"sse41", filmpi::Kernel::SSE41}, {"avx2", filmpi::Kernel::AVX2},
        {"neon", filmpi::Kernel::NEON},
    };
    for (const auto& entry : table) {
        if (std::strcmp(name, entry.name) == 0) {
            *out = entry.kernel;
            return true;
        }
    }
    return false;
}

std::string lutNameFromPath(const char* path) {
    const char* base = std::strrchr(path, '/');
    base = base ? base + 1 : path;
    const char* ext = std::strrchr(base, '.');
    return ext ? std::string(base, ext - base) : std::string(base);
}

bool isJpegPath(const char* path) {
    const char* ext = std::strrchr(path, '.');
    return ext && (std::strcmp(ext, ".jpg") == 0 || std::strcmp(ext, ".jpeg") == 0);
}

}  // namespace

int main(int argc, char** argv) {
    int quality = 95;
    int threads = 0;
    bool verbose = false;
    bool keepMetadata = true;
    filmpi::Kernel kernel = filmpi::Kernel::Auto;
    std::vector<const char*> positional;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        auto nextValue = [&]() -> const char* {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };

        if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (std::strcmp(arg, "--version") == 0) {
            std::printf("hald_clut %s\n", FILMPI_VERSION);
            return 0;
        } else if (std::strcmp(arg, "-v") == 0 || std::strcmp(arg, "--verbose") == 0) {
            verbose = true;
        } else if (std::strcmp(arg, "-M") == 0 || std::strcmp(arg, "--no-metadata") == 0) {
            keepMetadata = false;
        } else if (std::strcmp(arg, "-q") == 0 || std::strcmp(arg, "--quality") == 0) {
            const char* v = nextValue();
            if (!v || (quality = std::atoi(v)) < 1 || quality > 100) {
                std::fprintf(stderr, "Error: --quality expects a value in 1..100\n");
                return 1;
            }
        } else if (std::strcmp(arg, "-t") == 0 || std::strcmp(arg, "--threads") == 0) {
            const char* v = nextValue();
            if (!v || (threads = std::atoi(v)) < 0) {
                std::fprintf(stderr, "Error: --threads expects a non-negative number\n");
                return 1;
            }
        } else if (std::strcmp(arg, "-k") == 0 || std::strcmp(arg, "--kernel") == 0) {
            const char* v = nextValue();
            if (!v || !parseKernel(v, &kernel)) {
                std::fprintf(stderr,
                             "Error: --kernel expects auto|scalar|sse41|avx2|neon\n");
                return 1;
            }
        } else if (arg[0] == '-' && arg[1] != '\0') {
            std::fprintf(stderr, "Error: unknown option '%s'\n", arg);
            printUsage(argv[0]);
            return 1;
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.size() != 3) {
        printUsage(argv[0]);
        return 1;
    }
    const char* lutPath = positional[0];
    const char* inputPath = positional[1];
    const char* outputPath = positional[2];

    std::string error;
    const filmpi::Image lutImage = filmpi::Image::load(lutPath, &error);
    if (!lutImage.valid()) {
        std::fprintf(stderr, "Error: could not load HALD CLUT '%s': %s\n",
                     lutPath, error.c_str());
        return 1;
    }

    const filmpi::HaldClut clut = filmpi::makeHaldClut(
        lutImage.pixels(), lutImage.width(), lutImage.height(), &error);
    if (!clut.valid()) {
        std::fprintf(stderr, "Error: '%s': %s\n", lutPath, error.c_str());
        return 1;
    }

    filmpi::Image image = filmpi::Image::load(inputPath, &error);
    if (!image.valid()) {
        std::fprintf(stderr, "Error: could not load image '%s': %s\n",
                     inputPath, error.c_str());
        return 1;
    }

    if (verbose) {
        std::printf("image : %dx%d\n", image.width(), image.height());
        std::printf("clut  : %d^3\n", clut.size);
    }

    filmpi::applyHaldClut(clut, image.pixels(), image.pixelCount(), threads,
                          kernel);

    if (verbose) {
        std::printf("kernel: %s\n",
                    filmpi::kernelName(filmpi::resolveKernel(kernel)));
    }

    if (!filmpi::saveImage(outputPath, image, quality, &error)) {
        std::fprintf(stderr, "Error: could not write '%s': %s\n", outputPath,
                     error.c_str());
        return 1;
    }

    if (keepMetadata && isJpegPath(outputPath)) {
        const std::string lutName = lutNameFromPath(lutPath);
        if (!filmpi::copyMetadata(inputPath, outputPath, lutName.c_str()) &&
            verbose) {
            std::printf("note  : metadata not copied (%s)\n",
                        filmpi::metadataSupported() ? "copy failed"
                                                    : "built without Exiv2");
        }
    }

    return 0;
}
