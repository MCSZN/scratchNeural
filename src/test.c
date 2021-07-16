#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <neuron.h>

bnn_data_t* read_csv(void) {
    FILE* csv = fopen("data/xor.csv", "r");
    bnn_data_t* data;
    data->shape = shape;
    data->input = inputs;
    data->output = labels;
    fclose(csv); 
    return data;
}

int main() {
    srand(time(NULL));
    bnn_data_t data = read_csv();
    printf("bnn_data shape %li %li\n", data.shape.n_features, data.shape.n_samples);
    printf("bnn_data %i\n", data.input[100]);
    return 0;
}