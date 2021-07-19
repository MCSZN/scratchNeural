#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <neuron.h>

bnn_data_t* read_xor_csv(void) {
    printf("reading the xor file now\n");
    FILE* csv = fopen("data/xor.csv", "r");
    if (csv == NULL) return NULL;
    bnn_data_t* data = init_bnn_data(1000, 2);

    size_t line_counter = 0;
    char buffer[6];
    // read the file and fill the arrays
    while(line_counter < 1000) {
        if (fgets(buffer, 6, csv) == NULL) break;
        data->data[line_counter][0] = (buffer[0] == '0') ? false : true;
        data->data[line_counter][1] = (buffer[2] == '0') ? false : true;
        data->labels[line_counter] = (buffer[4] == '0') ? false : true;
        line_counter++;
    }
    fclose(csv); 
    return data;
}

int main() {
    srand(time(NULL));
    printf("reading the file\n");
    bnn_data_t* data = read_xor_csv();
    printf("read the file\n");
    u64 layer_sizes[3] = {2, 3, 1};
    activation_t activations[3] = {RELU, SIGMOID, TANH};
    bnn_config_t config = {
        .layer_sizes=layer_sizes,
        .activations=activations,
        .n_layers=3
    };
    printf("initialized the config for bnn init\n");
    bnn_t* bnn = init_bnn(config);
    // remember to deallocate bnn_data_t
    printf("%p\n", (void*)&bnn);
    deinit_bnn_data(data, 1);
    deinit_bnn(bnn, 1);
    return 0;
}
