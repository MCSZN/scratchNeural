#include <stdint.h>
#include <stdbool.h>

#ifndef ARRAYS_H
#define ARRAYS_H

typedef struct sized_array_t {
    uint16_t length;
    float* data;
} sized_array_t;

uint8_t rand_bit();

uint8_t* randbool(const uint64_t length);
float* randfloat(const uint64_t length);

uint32_t dot_product(uint8_t a[], uint8_t b[], uint16_t length);
uint32_t input_dot_product(float a[], uint8_t b[], uint16_t length);

#endif