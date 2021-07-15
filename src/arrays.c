#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <assert.h>


uint8_t rand_bit(const uint8_t n) {
    // return random number between 0 and n
    // n should not exceed 255
    return (rand() % n);
}

static inline uint8_t* rand_array(const uint64_t length, const uint8_t max_size) {
    // allocates array of size length
    // fills it with random numbers up to max size
    // remember to free the array !!
    uint8_t* data = (uint8_t*)malloc(length * sizeof(uint8_t));
    for (uint64_t i = 0; i < length; i++) {
        data[i] = rand_bit(max_size);
    }
    return data;
}

uint8_t* rand_bool(const uint64_t length) {
    // returns a array filled with random zeros and ones
    return rand_array(length, 2);
}

float* rand_float(const uint64_t length) {
    // returns array filled with values between 0 and 1
    uint8_t* arr = rand_array(length, 255);
    float* f_arr = (float*) malloc(sizeof(float) * length);
    for (uint8_t i = 0; i < length; i++) {
        f_arr[i] = (float) arr[i] / 255;
    }
    free(arr);
    return f_arr;
}

uint32_t dot_product(uint8_t a[], uint8_t b[], uint16_t length) {
    // dot product for regular one dimensional arrays
    uint32_t result = 0;
    for (uint16_t i = 0; i < length; i++) {
        result += a[i] * b[i];
    }
    return result;
}