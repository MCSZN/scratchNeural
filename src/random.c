#include <math.h>
#include <stdlib.h>
#include <numbers.h>


u8 rand_u8() {
    // return random u8
    return (u8)(rand() % 255);
}

bool rand_bool() {
    // returns a random boolean
    if (rand() % 2 == 0) return false;
    return true;
}

float rand_f32() {
    return (f32)(rand() / RAND_MAX);
}
