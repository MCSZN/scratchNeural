#include <stdio.h>
#include <numbers.h>
#include <neuron.h>
#include <random.h>
#include <math.h>

u16 dot_product(const bool* a, const bool* b, const size_t length) {
    // dot product for regular one dimensional arrays
    u16 result = 0;
    for (size_t i = 0; i < length; i++) {
        result += a[i] * b[i];
    }
    return result;
}

bnn_data_t* init_bnn_data(size_t n_samples, size_t n_features) {
    bnn_data_t* data = malloc(sizeof(bnn_data_t));
    if (data == NULL) return NULL;
    data->n_samples = n_samples;
    data->n_features = n_features;
    // allocate 1000 array of pointers for inputs and outputs
    data->data = malloc(sizeof(bool*) * n_samples);
    if (data->data == NULL) return NULL;
    data->labels = malloc(sizeof(bool) * n_samples);
    if (data->labels == NULL) return deinit_bnn_data(data, 2);
    // allocate each array now
    for (size_t i = 0; i < n_samples; i++) {
        data->data[i] = malloc(sizeof(bool) * n_features);
        if (data->data[i] == NULL) return deinit_bnn_data(data, 3);
    }
    return data;
}

void* deinit_bnn_data(bnn_data_t* data, u8 level) {
    // free allocated data
    if (level > 2) {
        for (int i = 0; i < 1000; i++) {
            free(data->data[i]);
        }
    } 
    if (level > 1) {
        free(data->labels);
    } 
    if (level > 0) {
        free(data->data);
    }
    return NULL;
}

bnn_t* init_bnn(bnn_config_t config) {
    // malloc bnn struct
    bnn_t* bnn = malloc(sizeof(bnn_t) * 1);
    if (bnn == NULL) return NULL;
    // init constants
    bnn->n_layers = config.n_layers;
    bnn->initialized = true;
    bnn->layers = malloc(sizeof(layer_t)* bnn->n_layers);
    // if (bnn->layers == NULL) deinit_bnn_t(bnn, 1);
    for (size_t i= 0; i < config.n_layers; i++) {
        // for each layer
        layer_t layer = bnn->layers[i];
        layer.activation = config.activations[i];
        layer.n_neurons = config.layer_sizes[i];
        layer.n_prev = (i != 0) ? config.layer_sizes[i-1] : config.layer_sizes[0];
        layer.bool_weights = malloc(sizeof(bool*) * layer.n_neurons);
        layer.float_weights = malloc(sizeof(f32*) * layer.n_neurons);
        for (size_t i = 0; i < layer.n_prev; i++) {
            // for each neuron
            layer.bool_weights[i] = malloc(sizeof(bool) * layer.n_prev);
            layer.float_weights[i] = malloc(sizeof(f32) * layer.n_prev);
            for (size_t j = 0; j < layer.n_prev; j++) {
                // for each weight
                layer.float_weights[i][j] = rand_f32();
                layer.bool_weights[i][j] = rand_bool();
            }
        }
    }
    return bnn;
}

void* deinit_bnn(bnn_t* bnn, u8 level) {
    return NULL;
}

bool sigmoid(const u16 input) {
    const f32 result = 1 / (1 + exp(-input));
    if (result > 0.5) return true;
    return false;
}

f32 sigmoid_derivative(const u16 input) {
    return sigmoid(input) * (1 - sigmoid(input));
}
