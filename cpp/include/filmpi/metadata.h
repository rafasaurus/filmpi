#ifndef FILMPI_METADATA_H
#define FILMPI_METADATA_H

namespace filmpi {

// True when the build includes Exiv2 metadata support.
bool metadataSupported();

// Copies EXIF/IPTC/XMP metadata and APP2 (ICC/MPF) segments from
// sourcePath to destPath, tagging the image description with lutName
// when non-empty. Returns false if metadata support is compiled out
// or copying fails; the pixel data of destPath is never touched.
bool copyMetadata(const char* sourcePath, const char* destPath,
                  const char* lutName);

}  // namespace filmpi

#endif  // FILMPI_METADATA_H
