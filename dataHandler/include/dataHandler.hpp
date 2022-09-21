#pragma once

#include <fstream>
#include <stdint.h>
#include "data.hpp"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <unordered_set>

class DataHandler
{
    std::vector<Data *> *dataArray;
    std::vector<Data *> *trainingData;
    std::vector<Data *> *testData;
    std::vector<Data *> *validationData;

    int numClasses;
    int featureVectorSize;
    std::map<uint8_t, int> classFromInt;

    constexpr static auto TRAINING_DATA_PERCENT = 0.80;
    constexpr static auto TEST_DATA_PERCENT = 0.20;
    constexpr static auto VAL_DATA_PERCENT = 0.0;

public:
    DataHandler();
    ~DataHandler();

    void readInputData(std::string path);
    void readLabelData(std::string path);
    void splitData();
    void countClasses();

    uint32_t convertToLittleEndian(const unsigned char *bytes);

    std::vector<Data *> *getDataArray() { return dataArray; }
    std::vector<Data *> *getTrainingData() { return trainingData; }
    std::vector<Data *> *getTestData() { return testData; }
    std::vector<Data *> *getValData() { return validationData; }
};
