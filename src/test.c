#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <neuron.h>

bnn_data_t read_csv(void) {
    FILE* csv = open("data/xor.csv", 'r');
    bool inputs[1000][2];
    bool labels[1000];
    bnn_data_t data; 
    fclose(csv);
    return data;
}

int main() {
    srand(time(NULL));
    bnn_data_t data = read_csv();
    return 0;
}