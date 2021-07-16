main:
	gcc --std=c11 -Wall --pedantic -O3 -o bin/test src/*.c -I headers -lm -g

assembly:
	gcc --std=c11 -Wall --pedantic -S src/test.c -o asm/main.asm -I headers -lm

clean:
	rm -rf bin/* && rm -rf asm/*

run:
	./bin/test