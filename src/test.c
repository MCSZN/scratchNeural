#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <neuron.h>

bnn_data_t* read_xor_csv(void) {
    FILE* csv = fopen("data/xor.csv", "r");
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
    bnn_data_t* data = read_xor_csv();
    // remember to deallocate bnn_data_t
    deinit_bnn_data(data);
    return 0;
}