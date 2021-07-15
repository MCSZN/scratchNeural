#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <activations.h>
#include <arrays.h>

typedef struct neuron_t {
    activation_t activation;
    sized_array_t weights;
    uint8_t bias;
} neuron_t;


neuron_t neuron_init(activation_t activation, uint8_t shape) {
    neuron_t neuron = {
        .activation = activation,
        .bias = rand_bit(2),
        .weights = randbool(shape)
    };
    return neuron;
}

void neuron_destroy(neuron_t neuron) {
    free(neuron.bias);
    free(neuron.weights.data);
}


int main() {
    srand(time(NULL));

    float* input_data= randfloat(10);
    neuron_t neuron = neuron_init(RELU, 10);

    free(input_data);
    return 0;
}