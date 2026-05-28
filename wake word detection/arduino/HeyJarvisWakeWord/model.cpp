// Placeholder model array.
//
// This file will be overwritten by ``python/convert.py`` after you have
// trained the wake-word model.  Until then the firmware will compile
// but inference will fail at runtime with a clear error message — which
// is the intended behaviour for a freshly-checked-out repository.
#include "model.h"

alignas(16) const unsigned char g_model[] = {
  // Empty placeholder — replace by running:
  //   cd python && python convert.py
  0x00
};

const unsigned int g_model_len = 1;
