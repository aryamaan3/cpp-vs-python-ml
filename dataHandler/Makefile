CC=g++
INCLUDE_DIR := $(MNIST_ROOT)/include
SRC := $(MNIST_ROOT)/src
CFLAGS := -std=c++17 -g
LIBRARY := libdata.so

all: $(LIBRARY)

$(LIBRARY): libdir objdir obj/dataHandler.o obj/data.o
	$(CC) $(CFLAGS) -o $(MNIST_ROOT)/lib/$(LIBRARY) obj/*.o
	rm -r $(MNIST_ROOT)/obj

libdir:
	mkdir -p $(MNIST_ROOT)/lib

objdir:
	mkdir -p $(MNIST_ROOT)/obj

obj/dataHandler.o: $(SRC)/dataHandler.cpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -c $(SRC)/dataHandler.cpp -o obj/dataHandler.o

obj/data.o: $(SRC)/data.cpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -c $(SRC)/data.cpp -o obj/data.o

clean:
	rm -r $(MNIST_ROOT)/lib
	rm -r $(MNIST_ROOT)/obj