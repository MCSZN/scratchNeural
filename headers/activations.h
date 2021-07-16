#include <stdbool.h>

#pragma once

typedef enum activation_t {
    RELU,
    SIGMOID,
    TANH
} activation_t;

bool sigmoid(uint16_t input, size_t len);