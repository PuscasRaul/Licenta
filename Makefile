all: build/ndarray.o build/logger.o

build/ndarray.o: src/Licenta/math/ndarray.c src/Licenta/math/ndarray.h
	cc -c src/Licenta/math/ndarray.c -o build/ndarray.o

build/logger.o: src/Licenta/Logger/Logger.c src/Licenta/Logger/Logger.h
	cc -c src/Licenta/Logger/Logger.c -o build/logger.o

tests: 

build/tests/ndarray_test.o: src/tests/ndarray.c
