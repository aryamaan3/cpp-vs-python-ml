#pragma once

#include <vector>
#include <data.hpp>
#include <memory>

class KNN
{
    int k;
    std::vector<Data *> *neighbors;
    std::vector<Data *> *trainingData;
    std::vector<Data *> *testData;

public:
    KNN(int _k, std::vector<Data *> *_trainingData, std::vector<Data *> *_testData) : k(_k), trainingData(_trainingData), testData(_testData){};
    KNN() = default;
    ~KNN() = default;

    void findKNearest(Data *queryPoint);
    void setK(int val);
    int predict();
    double calculateDistance(Data *queryPoint, Data *input);
    double test(int nbOfTest);
};
