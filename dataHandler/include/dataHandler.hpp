#pragma once

#include <fstream>
#include <stdint.h>
#include "data.hpp"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <unordered_set>

class dataHandler
{
    std::shared_ptr<std::vector<std::shared_ptr<data>>> _dataArray;
    std::shared_ptr<std::vector<std::shared_ptr<data>>> _trainingData;
    std::shared_ptr<std::vector<std::shared_ptr<data>>> _testData;
    std::shared_ptr<std::vector<std::shared_ptr<data>>> _valData;

    int _numClasses;
    int _featureVectorSize;
    std::map<uint8_t, int> _labelMap;

    constexpr static auto TRAINING_DATA_PERCENT = 0.80;
    constexpr static auto TEST_DATA_PERCENT = 0.10;
    constexpr static auto VAL_DATA_PERCENT = 0.10;

public:
    dataHandler();
    ~dataHandler();

    void readFeatureVector(std::string path);
    void readLabel(std::string path);
    void splitData();
    void countClasses();

    uint32_t convertToLittleEndian(const unsigned char *bytes);

    std::shared_ptr<std::vector<std::shared_ptr<data>>> getTrainingData();
    std::shared_ptr<std::vector<std::shared_ptr<data>>> getTestData();
    std::shared_ptr<std::vector<std::shared_ptr<data>>> getValData();
};
