#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <neuron.h>
#include <math.h>

uint16_t dot_product(const bool* a, const bool* b, const size_t length) {
    // dot product for regular one dimensional arrays
    uint16_t result = 0;
    for (size_t i = 0; i < length; i++) {
        result += a[i] * b[i];
    }
    return result;
}

bnn_data_t* init_bnn_data(size_t n_samples, size_t n_features) {
    bnn_data_t* data = malloc(sizeof(bnn_data_t));
    data->n_samples = n_samples;
    data->n_features = n_features;
    // allocate 1000 array of pointers for inputs and outputs
    data->data = malloc(sizeof(bool*) * n_samples);
    data->labels = malloc(sizeof(bool) * n_samples);
    // allocate each array now
    for (size_t i = 0; i < n_samples; i++) {
        data->data[i] = malloc(sizeof(bool) * n_features);
    }
    return data;
}

void deinit_bnn_data(bnn_data_t* data) {
    // free allocated data
    for (int i = 0; i < 1000; i++) {
        free(data->data[i]);
    }
    free(data->data);
    free(data->labels);
}

bool sigmoid(const uint16_t input) {
    const float result = 1 / (1 + exp(-input));
    if (result > 0.5) return true;
    return false;
}

float sigmoid_derivative(const uint16_t input) {
    return sigmoid(input) * (1 - sigmoid(input));
}

void train(bnn_t* bnn, bnn_train_config_t config) {
    if (!bnn->initialized) return;
    // begin training
    for (size_t i = 0; i < config.epochs; i++) {
        // forward pass
        for (size_t j = 0; j < bnn->n_layers; j++) {
            layer_t layer = bnn->layers[j];
            // create zero initialized array
            for (size_t k = 0; k < layer.n_neurons; k++) {
                neuron_t neuron = layer.neurons[k];
                //uint16_t output = dot_product(neuron.weights_bool, );
                if (config.verbose) {
                    printf("forward pass: layer %li neuron %li value %i", j, k, 0);
                }
            }
        }
    }
}