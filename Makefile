CC      := gcc
CFLAGS  := -g -Wall

SRC         := src
BUILD       := build
SRC_LIB     := $(SRC)/Licenta
SRC_TEST    := $(SRC)/tests
BUILD_TEST  := $(BUILD)/tests

.PHONY: all test clean

all: $(BUILD) $(BUILD_TEST) $(BUILD)/ndarray.o $(BUILD)/logger.o test

test: $(BUILD_TEST)/ndarray_test $(BUILD_TEST)/arena_test
	@echo "Running tests..."
	./$(BUILD_TEST)/ndarray_test
	#./$(BUILD_TEST)/arena_test

$(BUILD_TEST)/ndarray_test: $(SRC_TEST)/ndarray.c $(BUILD)/ndarray.o
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD_TEST)/arena_test: $(SRC_TEST)/arena_allocator.c $(SRC_LIB)/utils/arena.h
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD)/ndarray.o: $(SRC_LIB)/math/ndarray.c $(SRC_LIB)/math/ndarray.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/logger.o: $(SRC_LIB)/Logger/Logger.c $(SRC_LIB)/Logger/Logger.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD_TEST):
	mkdir -p $(BUILD_TEST)

clean:
	rm -f $(BUILD)/*.o $(BUILD_TEST)/* build/test_runner

