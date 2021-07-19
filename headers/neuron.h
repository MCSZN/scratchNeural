#include <stdlib.h>
#include <numbers.h>

#pragma once


u16 dot_product(const bool* a, const bool* b, const size_t length);

typedef enum activation_t {
    RELU,
    SIGMOID,
    TANH
} activation_t;

typedef struct layer_t {
    // each neuron (n) has (m) weights 
    // corresponding to the number of neurons in the previous layer
    bool** bool_weights;
    f32** float_weights;
    size_t n_neurons;
    size_t n_prev;
    activation_t activation;
} layer_t;

typedef struct bnn_t {
    layer_t* layers;
    size_t n_layers;
    bool initialized;
} bnn_t;

typedef struct bnn_config_t {
    size_t n_layers;
    u64* layer_sizes;
    activation_t* activations;
} bnn_config_t;

typedef struct bnn_train_config_t {
    size_t epochs;
    bool verbose;
} bnn_train_config_t;


typedef struct bnn_data_t {
    bool** data;
    bool* labels; 
    size_t n_samples;
    size_t n_features;
} bnn_data_t;

bnn_data_t* init_bnn_data(size_t n_samples, size_t n_features);
void* deinit_bnn_data(bnn_data_t* data, u8 level);

bnn_t* init_bnn(bnn_config_t config);
void* deinit_bnn(bnn_t* bnn, u8 level);

void train(bnn_t* bnn, bnn_train_config_t config);
