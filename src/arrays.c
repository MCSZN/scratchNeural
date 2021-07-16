#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>


uint8_t rand_uint8() {
    // return random u8
    return (uint8_t)(rand() % 255);
}

bool rand_bool() {
    // returns a random boolean
    if (rand() % 2 == 0) return false;
    return true;
}

float rand_float() {
    return (float)(rand_uint8(255)) / 255;
}

uint16_t dot_product(const bool a[], const bool b[], const size_t length) {
    // dot product for regular one dimensional arrays
    uint16_t result = 0;
    for (size_t i = 0; i < length; i++) {
        result += a[i] * b[i];
    }
    return result;
}