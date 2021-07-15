#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <activations.h>


float sigmoid(float input) {
    return 1 / (1 + exp(-input));
}

float relu(float input) {
    return (input > 0.0) ? input : 0.0;
}
