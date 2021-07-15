#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

typedef enum activation_t {
    RELU,
    SIGMOID,
    TANH
} activation_t;

float sigmoid(float input);
float relu(float input);

#endif