#include "knn.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "dataHandler.hpp"
#include <memory>
#include <iostream>
#include <chrono>

using namespace std::chrono;

void KNN::findKNearest(Data *queryPoint)
{
    neighbors = new std::vector<Data*>;
    auto min = std::numeric_limits<double>::max();
    auto previousMin = min;
    int index;
    for (auto i = 0; i < k; i++)
    {
        if (i == 0)
        {
            for (auto j = 0; j < trainingData->size(); j++)
            {
                double dist = calculateDistance(queryPoint, trainingData->at(j));
                trainingData->at(j)->setDistance(dist);
                if (dist < min)
                {
                    min = dist;
                    index = j;
                }
            }
            neighbors->push_back(trainingData->at(index));
            previousMin = min;
            min = std::numeric_limits<double>::max();
        }
        else
        {
            for (int j = 0; j < trainingData->size(); j++)
            {
                double dist = trainingData->at(j)->getDistance();
                if (dist > previousMin && dist < min)
                {
                    min = dist;
                    index = j;
                }
            }
            neighbors->push_back(trainingData->at(index));
            previousMin = min;
            min = std::numeric_limits<double>::max();
        }
    }
}
void KNN::setK(int val)
{
    k = val;
}

int KNN::predict()
{
    std::map<uint8_t, int> frequencyMap;
    for (int i = 0; i < neighbors->size(); i++)
    {
        if (frequencyMap.find(neighbors->at(i)->getLabel()) == frequencyMap.end())
        {
            frequencyMap[neighbors->at(i)->getLabel()] = 1;
        }
        else
        {
            frequencyMap[neighbors->at(i)->getLabel()]++;
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
    double value = 0;
    if (queryPoint->getFeatureVectorSize() != input->getFeatureVectorSize())
    {
        printf("Vector size mismatch.\n");
        exit(1);
    }

    for (unsigned i = 0; i < queryPoint->getFeatureVectorSize(); i++)
    {
        value += pow(queryPoint->getFeatureVector()->at(i) - input->getFeatureVector()->at(i), 2);
    }

    return sqrt(value);
}

double KNN::test(int nbOfTest)
{
    std::cout << "------------------\n";
    std::cout << "Testing slow...\n";

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

    std::cout << "Correct: " << correct << " out of " << nbOfTest << "\n";
    std::cout << "Accuracy: " << (double)correct / nbOfTest * 100 << "%\n";
    std::cout << "------------------\n";
    std::cout << "Total time: " << (double)sumOfTimes / 1000000 << " s\n";
    std::cout << "Average time: : " << (double)((sumOfTimes / nbOfTest) / 1000000) << " s\n";

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
    model->test(10);
}
