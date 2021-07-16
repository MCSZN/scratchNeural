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
    return (float)(rand() / RAND_MAX);
}