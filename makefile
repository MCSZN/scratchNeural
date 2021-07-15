main:
	gcc -Wall --pedantic -o bin/main src/*.c -I headers -lm

assembly:
	gcc -Wall --pedantic -S -fverbose-asm src/main.c -o asm/main.asm -I headers -lm

clean:
	rm -rf bin/* && rm -rf asm/*

run:
	./bin/main