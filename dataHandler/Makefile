CC=g++
INCLUDE_DIR := $(MNIST_ROOT)/dataHandler/include
SRC := $(MNIST_ROOT)/dataHandler/src
CFLAGS := -shared -std=c++17 -g
LIBRARY := libdata.so

all: $(LIBRARY)

$(LIBRARY): libdir objdir obj/dataHandler.o obj/data.o
	$(CC) $(CFLAGS) -o $(MNIST_ROOT)/dataHandler/lib/$(LIBRARY) obj/*.o
	rm -r $(MNIST_ROOT)/dataHandler/obj

libdir:
	mkdir -p $(MNIST_ROOT)/dataHandler/lib

objdir:
	mkdir -p $(MNIST_ROOT)/dataHandler/obj

obj/dataHandler.o: $(SRC)/dataHandler.cpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -c $(SRC)/dataHandler.cpp -o obj/dataHandler.o

obj/data.o: $(SRC)/data.cpp
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -c $(SRC)/data.cpp -o obj/data.o

clean:
	rm -r $(MNIST_ROOT)/dataHandler/lib
	rm -r $(MNIST_ROOT)/dataHandler/obj