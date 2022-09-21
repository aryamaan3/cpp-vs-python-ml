#pragma once

#include <vector>
#include <data.hpp>
#include <memory>

class knn
{
    int _k;
    std::shared_ptr<std::vector<std::shared_ptr<data>>> _neighbours;
    std::shared_ptr<std::vector<std::shared_ptr<data>>> _trainingData;
    std::shared_ptr<std::vector<std::shared_ptr<data>>> _testData;

    public:

    knn(int k, std::shared_ptr<std::vector<std::shared_ptr<data>>> trainingData, std::shared_ptr<std::vector<std::shared_ptr<data>>> testData);
    knn() = default;
    ~knn() = default;

    void findKNearest(std::shared_ptr<data> d);
    void findKNearestFast(std::shared_ptr<data> d);
    void setTrainingData(std::shared_ptr<std::vector<std::shared_ptr<data>>> trainingData);
    void setTestData(std::shared_ptr<std::vector<std::shared_ptr<data>>> testData);
    void setK(int k);

    int predict();
    double calculateDistance(std::shared_ptr<data> d1, std::shared_ptr<data> d2);
    double test(int nbOfTest);
    double testFast(int nbOfTest);

};

