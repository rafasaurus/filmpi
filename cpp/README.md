# filmpi — HALD CLUT applier

Applies a [HALD CLUT](https://www.quelsolaar.com/technology/clut.html) to an
image. Trilinear interpolation with SIMD kernels (AVX2 / SSE4.1 on x86,
NEON on ARM), multithreaded, with EXIF/IPTC/XMP and ICC-profile
preservation for JPEG output (via Exiv2, optional).

## Layout

```
cpp/
├── CMakeLists.txt        platform-autodetecting build
├── cli/main.cpp          hald_clut command line tool
├── include/filmpi/       public library API (clut.h, image.h, metadata.h)
├── src/
│   ├── clut.cpp          CLUT construction, runtime kernel dispatch, threading
│   ├── image.cpp         stb-based image I/O
│   ├── metadata.cpp      Exiv2 metadata + APP2 segment copying
│   └── kernels/          one trilinear kernel per instruction set
└── external/stb/         vendored stb headers
```

## Building

```sh
cmake -S . -B build
cmake --build build -j
cmake --build build --target install     # installs to ~/.local/bin
cmake --build build --target uninstall
```

The configure step prints a summary of what was autodetected:

```
filmpi 1.0.0 configuration:
  target         : Linux / x86_64
  SIMD kernels   : scalar, sse4.1, avx2
  exiv2 metadata : TRUE
  install prefix : /home/you/.local
```

All compiled kernels are checked against the CPU at runtime, so a binary
built on one x86 machine picks the right kernel on another.

### Options

| Option              | Default | Effect                                        |
|---------------------|---------|-----------------------------------------------|
| `FILMPI_SIMD`       | `ON`    | build SIMD kernels for the target architecture |
| `FILMPI_NATIVE`     | `OFF`   | add `-march=native` for this machine           |
| `FILMPI_WITH_EXIV2` | `AUTO`  | metadata support (`AUTO`/`ON`/`OFF`)           |
| `FILMPI_BUILD_CLI`  | `ON`    | build the `hald_clut` executable               |

### Raspberry Pi / Jetson

Native builds need nothing special — NEON is detected from the target
processor (always on for the 64-bit OS, runtime-checked via `getauxval`
on 32-bit Raspberry Pi OS):

```sh
sudo apt install cmake g++ libexiv2-dev   # exiv2 optional
cmake -S . -B build && cmake --build build -j
```

### Android (NDK)

```sh
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24
cmake --build build-android -j
```

Exiv2 is not available there by default, so metadata support is compiled
out automatically (`FILMPI_WITH_EXIV2=AUTO`); pixel processing is
unaffected. Link `libfilmpi.a` from JNI, or push `hald_clut` with adb.

### Cross-compiling for ARM Linux

```sh
cmake -S . -B build-arm64 \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++
```

## Usage

```
hald_clut [options] <hald_clut> <input_image> <output_image>

  -q, --quality <1-100>  JPEG quality (default 95)
  -t, --threads <n>      worker threads (default: all cores)
  -k, --kernel <name>    auto|scalar|sse41|avx2|neon (default auto)
  -M, --no-metadata      do not copy EXIF/IPTC/XMP metadata
  -v, --verbose          print kernel and image details
```

Output format follows the output extension: `.jpg`/`.jpeg` writes JPEG
(metadata copied from the input, LUT name written to the image
description), anything else writes PNG.

## Library API

```cpp
#include <filmpi/clut.h>
#include <filmpi/image.h>

filmpi::Image lut = filmpi::Image::load("velvia.png");
filmpi::HaldClut clut = filmpi::makeHaldClut(lut.pixels(), lut.width(), lut.height());

filmpi::Image photo = filmpi::Image::load("photo.jpg");
filmpi::applyHaldClut(clut, photo.pixels(), photo.pixelCount());  // in place
filmpi::saveImage("out.jpg", photo, 95);
```

## Performance

12-megapixel image, level-8 (64³) CLUT, i7 with AVX2, 12 threads:

| Kernel            | Time    |
|-------------------|---------|
| scalar, 1 thread  | ~400 ms |
| SSE4.1, 1 thread  | ~237 ms |
| AVX2, 1 thread    | ~181 ms |
| AVX2, all cores   | ~22 ms  |

All kernels produce byte-identical output; an identity CLUT reproduces
the input image exactly.
