#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <activations.h>


bool sigmoid(const uint16_t input,const size_t len) {
    float result = 1 / (1 + exp(-(input/len)));
    if (result > 0.5) return true;
    return false;
}
