#include "ggml-sycl.h"

#include <cstdio>

extern "C" void ggml_cycl_init() {
    printf("ggml_cycl_init\n");
}