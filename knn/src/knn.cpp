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

knn::knn(int k, std::shared_ptr<std::vector<std::shared_ptr<data>>> trainingData, std::shared_ptr<std::vector<std::shared_ptr<data>>> testData)
{
    _k = k;
    _trainingData = trainingData;
    _testData = testData;
}

void knn::findKNearest(std::shared_ptr<data> d)
{
    for (auto &i : *_trainingData) // complexity = O(n^2) if k is small, O(n^3) if k is large
    {
        i->setDistance(calculateDistance(d, i));
    }

    std::sort(_trainingData->begin(), _trainingData->end(), [](std::shared_ptr<data> a, std::shared_ptr<data> b)
              { return a->getDistance() < b->getDistance(); }); // complexity = O(n log n)

    _neighbours = std::make_shared<std::vector<std::shared_ptr<data>>>(_trainingData->begin(), _trainingData->begin() + _k);
}

void knn::findKNearestFast(std::shared_ptr<data> d)
{
    _neighbours = std::make_shared<std::vector<std::shared_ptr<data>>>();
    auto min = std::numeric_limits<double>::max();
    auto prevMin = min;
    auto index = 0;

    for (auto i = 0U; i < _k; ++i)
    {
        if (i == 0)
        {
            for (auto j = 0U; j < _trainingData->size(); ++j)
            {
                auto distance = calculateDistance(d, _trainingData->at(j));
                _trainingData->at(j)->setDistance(distance);
                if (distance < min)
                {
                    min = distance;
                    index = j;
                }
            }
            _neighbours->push_back(_trainingData->at(index));
            prevMin = min;
            min = std::numeric_limits<double>::max();
        }
        else
        {
            for (auto j = 0U; j < _trainingData->size(); ++j)
            {
                auto distance = _trainingData->at(j)->getDistance();
                if (distance < min && distance > prevMin)
                {
                    min = distance;
                    index = j;
                }
            }
            _neighbours->push_back(_trainingData->at(index));
            prevMin = min;
            min = std::numeric_limits<double>::max();
        }
    }
}

void knn::setTrainingData(std::shared_ptr<std::vector<std::shared_ptr<data>>> trainingData)
{
    _trainingData = trainingData;
}

void knn::setTestData(std::shared_ptr<std::vector<std::shared_ptr<data>>> testData)
{
    _testData = testData;
}

void knn::setK(int k)
{
    _k = k;
}

int knn::predict()
{
    std::map<uint8_t, int> labelCount;
    for (auto &aNeighbour : *_neighbours)
    {
        if (labelCount.find(aNeighbour->getLabel()) == labelCount.end())
        {
            labelCount[aNeighbour->getLabel()] = 1;
        }
        else
        {
            labelCount[aNeighbour->getLabel()]++;
        }
    }

    auto max = 0;
    auto bestLabel = 0;

    for (auto &aLabel : labelCount)
    {
        if (aLabel.second > max)
        {
            max = aLabel.second;
            bestLabel = aLabel.first;
        }
    }

    return bestLabel;
}

double knn::calculateDistance(std::shared_ptr<data> d1, std::shared_ptr<data> d2)
{
    if (d1->getFeatureVectorSize() != d2->getFeatureVectorSize())
    {
        std::cout << "Error: Feature vector sizes do not match\n";
        exit(1);
    }

    double distance = 0.0;
    for (auto i = 0U; i < d1->getFeatureVectorSize(); ++i)
    {
        distance += pow(d1->getFeatureVector()->at(i) - d2->getFeatureVector()->at(i), 2);
    }

    return sqrt(distance);
}

double knn::test(int nbOfTest)
{
    std::cout << "------------------\n";
    std::cout << "Testing slow...\n";

    std::vector<double> times;
    double sumOfTimes = 0.0;

    auto correct = 0;
    for (auto i = 0; i < nbOfTest; ++i)
    {
        auto start = high_resolution_clock::now();

        findKNearest(_testData->at(i));
        auto prediction = predict();

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        times.push_back(duration.count());
        sumOfTimes += duration.count();

        if (prediction == (int)_testData->at(i)->getLabel())
        {
            correct++;
        }
    }

    std::cout << "Total time: " << sumOfTimes << " microseconds\n";
    std::cout << "Average time: " << sumOfTimes / nbOfTest << " microseconds\n";
    std::cout << "Correct: " << correct << " out of " << nbOfTest << "\n";
    std::cout << "Accuracy: " << (double)correct / nbOfTest * 100 << "%\n";
    std::cout << "------------------\n";

    return (double)correct / nbOfTest;
}

double knn::testFast(int nbOfTest)
{
    std::cout << "------------------\n";
    std::cout << "Testing fast...\n";

    std::vector<double> times;
    double sumOfTimes = 0.0;

    auto correct = 0;
    for (auto i = 0; i < nbOfTest; ++i)
    {
        auto start = high_resolution_clock::now();

        findKNearestFast(_testData->at(i));
        auto prediction = predict();

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        times.push_back(duration.count());
        sumOfTimes += duration.count();

        if (prediction == _testData->at(i)->getLabel())
        {
            correct++;
        }
    }

    std::cout << "Total time: " << sumOfTimes << " microseconds\n";
    std::cout << "Average time: " << sumOfTimes / nbOfTest << " microseconds\n";
    std::cout << "Correct: " << correct << " out of " << nbOfTest << "\n";
    std::cout << "Accuracy: " << (double)correct / nbOfTest * 100 << "%\n";
    std::cout << "------------------\n";

    return (double)correct / nbOfTest;
}

int main()
{
    std::string trainFeaturePath = "../assets/train-images.idx3-ubyte";
    std::string trainLabelPath = "../assets/train-labels.idx1-ubyte";

    dataHandler dh;
    dh.readFeatureVector(trainFeaturePath);
    dh.readLabel(trainLabelPath);
    dh.countClasses();
    dh.splitData();

    auto model = std::make_unique<knn>(1, dh.getTrainingData(), dh.getTestData());
    model->test(10);
    model->testFast(10);
}
