#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <activations.h>
#include <arrays.h>


int main() {
    srand(time(NULL));
    bool b = rand_bool();
    uint8_t num = rand_uint8();
    return 0;
}