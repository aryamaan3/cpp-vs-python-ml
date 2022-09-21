#!/bin/bash

export MNIST_ROOT=$PWD

cd dataHandler
make clean
make

cd ../knn
make clean
make

./main
cd ..