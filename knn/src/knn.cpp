#include "knn.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "dataHandler.hpp"
#include <iostream>
#include <chrono>

using namespace std::chrono;

void KNN::findKNearest(Data *queryPoint)
{
    neighbors = new std::vector<Data *>();
    for (auto &data : *trainingData)
    {
        data->setDistance(calculateDistance(queryPoint, data));
    }

    std::sort(trainingData->begin(), trainingData->end(), [](Data *a, Data *b)
              { return a->getDistance() < b->getDistance(); });

    for (int i = 0; i < k; i++)
    {
        neighbors->push_back(trainingData->at(i));
    }
}
void KNN::setK(int val)
{
    k = val;
}

int KNN::predict()
{
    std::map<uint8_t, int> frequencyMap;
    for (auto aNeighbor : *neighbors)
    {
        if (frequencyMap.find(aNeighbor->getLabel()) == frequencyMap.end())
        {
            frequencyMap[aNeighbor->getLabel()] = 1;
        }
        else
        {
            frequencyMap[aNeighbor->getLabel()]++;
        }
    }

    int best = 0;
    int max = 0;

    for (auto kv : frequencyMap)
    {
        if (kv.second > max)
        {
            max = kv.second;
            best = kv.first;
        }
    }
    delete neighbors;
    return best;
}

double KNN::calculateDistance(Data *queryPoint, Data *input)
{
    auto value = 0.0;
    if (queryPoint->getFeatureVectorSize() != input->getFeatureVectorSize())
    {
        std::cout << ("Vector size mismatch.\n");
        exit(1);
    }

    for (auto i = 0U; i < queryPoint->getFeatureVectorSize(); i++)
    {
        value += pow(queryPoint->getFeatureVector()->at(i) - input->getFeatureVector()->at(i), 2);
    }

    return sqrt(value);
}

int KNN::predictOne(Data *queryPoint)
{
    findKNearest(queryPoint);
    return predict();
}

double KNN::test(int nbOfTest)
{
    std::cout << "\n------------------\n";

    std::vector<double> times;
    double sumOfTimes = 0.0;

    auto correct = 0;
    for (auto i = 0; i < nbOfTest; ++i)
    {
        auto start = high_resolution_clock::now();

        findKNearest(testData->at(i));
        auto prediction = predict();

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        times.push_back(duration.count());
        sumOfTimes += duration.count();

        if (prediction == (int)testData->at(i)->getLabel())
        {
            correct++;
        }
    }

    std::cout << "For k : " << k << ":\n";
    std::cout << "Accuracy: " << (double)correct / nbOfTest * 100 << "%\n";
    std::cout << "Total Time: " << (double)sumOfTimes / 1000000 << " s\n";
    std::cout << "Average Time: : " << (double)((sumOfTimes / nbOfTest) / 1000000) << " s\n";

    return (double)correct / nbOfTest;
}

int main()
{
    std::string trainFeaturePath = "../assets/train-images.idx3-ubyte";
    std::string trainLabelPath = "../assets/train-labels.idx1-ubyte";

    DataHandler dh;
    dh.readInputData(trainFeaturePath);
    dh.readLabelData(trainLabelPath);
    dh.countClasses();
    dh.splitData();

    auto model = std::make_unique<KNN>(1, dh.getTrainingData(), dh.getTestData());
    model->test(100);
}
