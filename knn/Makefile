CC=g++
INCLUDE_DIR := $(MNIST_ROOT)/knn/include
SRC := $(MNIST_ROOT)/knn/src
CFLAGS := -std=c++17 -g -O3

all: main

main: $(SRC)/knn.cpp
	$(CC) $(CFLAGS) $(SRC)/knn.cpp -o main -L$(MNIST_ROOT)/dataHandler/lib -I$(INCLUDE_DIR) \
	-I$(MNIST_ROOT)/dataHandler/include -ldata

clean:
	rm main
