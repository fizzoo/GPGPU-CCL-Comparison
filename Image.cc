#include "Image.h"

using namespace iml;

bool Image::loadpng(const std::string &filename) {
  // Open the file. libpng expects C file handle.
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    std::cerr << "Couldn't open image file." << std::endl;
    return false;
  }

  // Handles for settings for reading
  png_structp pngp =
      png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  png_infop pngi = png_create_info_struct(pngp);
  if (!pngp || !pngi) {
    std::cerr << "Couldn't create png/info struct" << std::endl;
    png_destroy_read_struct(&pngp, &pngi, nullptr);
    fclose(fp);
    return false;
  }

  // Associate with the file
  png_init_io(pngp, fp);

  // Start reading the info at the start of the file (height, width, etc)
  png_read_info(pngp, pngi);

  // Get the info we need
  _height = png_get_image_height(pngp, pngi);
  _width = png_get_image_width(pngp, pngi);
  if (_height <= 0 || _width <= 0) {
    std::cerr << "Found no image data, zero dimension" << std::endl;
    png_destroy_read_struct(&pngp, &pngi, nullptr);
    fclose(fp);
    return false;
  }

  // Try to expand to rgba. May change the space required.
  int bit_depth = png_get_bit_depth(pngp, pngi);
  int color_type = png_get_color_type(pngp, pngi);
  if (color_type == PNG_COLOR_TYPE_PALETTE) {
    png_set_palette_to_rgb(pngp);
  }
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
    png_set_expand_gray_1_2_4_to_8(pngp);
  }
  if (png_get_valid(pngp, pngi, PNG_INFO_tRNS)) {
    png_set_tRNS_to_alpha(pngp);
  }
  if (color_type == PNG_COLOR_TYPE_RGB) {
    png_set_filler(pngp, 255, PNG_FILLER_BEFORE);
  }

  // Get length of row for allocation, allocate
  int rowbytes = png_get_rowbytes(pngp, pngi);
  unsigned char **row_pointers = new unsigned char *[_height];
  data = new unsigned char[rowbytes * _height];

  // Have row_pointers point into data sequentially, read into data
  for (unsigned int i = 0; i < _height; ++i) {
    row_pointers[i] = data + i * rowbytes;
  }
  png_read_image(pngp, row_pointers);
  delete[] row_pointers;

  // Cleanup
  png_destroy_read_struct(&pngp, &pngi, nullptr);
  fclose(fp);

  return true;
}

Image::Image(const std::string &filename) {
  auto pos = filename.find(".png");
  if (pos == filename.length() - 4) {
    ok = loadpng(filename);
    return;
  }
}

Image::Image(size_t width, size_t height) : _width(width), _height(height){
  data = new unsigned char[width*height*4];
}

bool iml::writepng(const std::string &filename, Image *img) {
  return writepng(filename, img->width(), img->height(), img->data);
}

bool iml::writepng(const std::string &filename, size_t width, size_t height,
                   unsigned char *data) {
  FILE *fp = fopen(filename.c_str(), "wb");
  if (!fp) {
    return false;
  }

  png_structp pngp =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!pngp) {
    return false;
  }
  png_infop pngi = png_create_info_struct(pngp);
  if (!pngi) {
    png_destroy_write_struct(&pngp, (png_infopp)NULL);
    return false;
  }

  png_init_io(pngp, fp);

  png_set_IHDR(pngp, pngi, width, height, 8, PNG_COLOR_TYPE_RGB_ALPHA,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(pngp, pngi);

  // Have row_pointers point into data sequentially, write to file
  unsigned char **row_pointers = new unsigned char *[height];
  for (size_t i = 0; i < height; ++i) {
    row_pointers[i] = data + i * width * 4;
  }
  png_write_image(pngp, row_pointers);
  delete[] row_pointers;

  png_write_end(pngp, pngi);

  // Cleanup
  png_destroy_write_struct(&pngp, &pngi);

  return true;
}

Image::~Image() { delete[] data; }