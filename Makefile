CC      := gcc
CFLAGS  := -g -Wall

SRC     := src
BUILD   := build
SRC_LIB := $(SRC)/Licenta
SRC_MATH := $(SRC_LIB)/math
SRC_LOG  := $(SRC_LIB)/Logger
SRC_TEST := $(SRC)/tests
BUILD_TEST := $(BUILD)/tests

.PHONY: all test

all: $(BUILD)/ndarray.o $(BUILD)/logger.o test

test: $(BUILD_TEST)/ndarray_test
	@echo "Running tests..."
	./$(BUILD_TEST)/ndarray_test

$(BUILD_TEST)/ndarray_test: $(SRC_TEST)/ndarray.c $(BUILD)/ndarray.o
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD)/ndarray.o: $(SRC_MATH)/ndarray.c $(SRC_MATH)/ndarray.h
	$(CC) -c $(CFLAGS) $< -o $@

$(BUILD)/logger.o: $(SRC_LOG)/Logger.c $(SRC_LOG)/Logger.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f $(BUILD)/*.o $(BUILD_TEST)/* build/test_runner

