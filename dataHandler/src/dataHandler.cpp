#include "dataHandler.hpp"
#include <iostream>

constexpr static auto OFFSET_FEATURES = 4;
constexpr static auto OFFSET_LABELS = 2;

DataHandler::DataHandler()
{
    dataArray = new std::vector<Data *>();
    trainingData = new std::vector<Data *>();
    testData = new std::vector<Data *>();
    validationData = new std::vector<Data *>();
}

DataHandler::~DataHandler()
{
    // might be useless
}

void DataHandler::readInputData(std::string aPath)
{
    uint32_t header[OFFSET_FEATURES]; // MAGIC, NBITEMS, NBROWS, NBCOLS
    unsigned char bytes[OFFSET_FEATURES];
    FILE *file = fopen(aPath.c_str(), "r");

    if (file == NULL)
    {
        std::cout << "Could not open file " << aPath << "\n";
        exit(1);
    }

    for (auto i = 0; i < OFFSET_FEATURES; ++i)
    {
        fread(bytes, sizeof(bytes), 1, file);
        header[i] = convertToLittleEndian(bytes);
    }

    auto itemSize = header[2] * header[3];
    for (auto i = 0; i < header[1]; ++i)
    {
        auto d = new Data();
        uint8_t element[1];
        for (auto j = 0; j < itemSize; ++j)
        {
            fread(element, sizeof(element), 1, file);
            d->appendFeatureVector(element[0]);
        }
        dataArray->push_back(d);
    }

    std::cout << "Retrieved features, size = " << dataArray->size() << "\n";
}

void DataHandler::readLabelData(std::string aPath)
{
    uint32_t header[OFFSET_LABELS]; // MAGIC, NBITEMS
    unsigned char bytes[OFFSET_FEATURES];
    FILE *file = fopen(aPath.c_str(), "rb");

    if (file == NULL)
    {
        std::cout << "Could not open file " << aPath << "\n";
        exit(1);
    }

    for (auto i = 0; i < OFFSET_LABELS; ++i)
    {
        fread(bytes, sizeof(bytes), 1, file);
        header[i] = convertToLittleEndian(bytes);
    }

    for (auto i = 0; i < header[1]; ++i)
    {
        uint8_t element[1];
        fread(element, sizeof(element), 1, file);
        dataArray->at(i)->setLabel(element[0]);
    }

    std::cout << "Retrieved labels, size = " << dataArray->size() << "\n";
}

void DataHandler::splitData()
{
    std::unordered_set<int> usedIndexes;
    auto trainingDataSize = dataArray->size() * TRAINING_DATA_PERCENT;
    auto testDataSize = dataArray->size() * TEST_DATA_PERCENT;
    auto valDataSize = dataArray->size() * VAL_DATA_PERCENT;

    for (auto i = 0; i < trainingDataSize; ++i)
    {
        auto index = rand() % dataArray->size();
        while (usedIndexes.find(index) != usedIndexes.end())
        {
            index = rand() % dataArray->size();
        }
        usedIndexes.insert(index);
        trainingData->push_back(dataArray->at(index));
    }

    for (auto i = 0; i < testDataSize; ++i)
    {
        auto index = rand() % dataArray->size();
        while (usedIndexes.find(index) != usedIndexes.end())
        {
            index = rand() % dataArray->size();
        }
        usedIndexes.insert(index);
        testData->push_back(dataArray->at(index));
    }

    for (auto i = 0; i < valDataSize; ++i)
    {
        auto index = rand() % dataArray->size();
        while (usedIndexes.find(index) != usedIndexes.end())
        {
            index = rand() % dataArray->size();
        }
        usedIndexes.insert(index);
        validationData->push_back(dataArray->at(index));
    }

    std::cout << "Training data size = " << trainingData->size() << "\n";
    std::cout << "Test data size = " << testData->size() << "\n";
    std::cout << "Val data size = " << validationData->size() << "\n";
}

void DataHandler::countClasses()
{
    auto count = 0;
    for (auto i = 0U; i < dataArray->size(); ++i)
    {
        if (classFromInt.find(dataArray->at(i)->getLabel()) == classFromInt.end())
        {
            classFromInt[dataArray->at(i)->getLabel()] = count;
            dataArray->at(i)->setEnumLabel(count);
            ++count;
        }
    }
    numClasses = count;
    std::cout << "Number of classes = " << numClasses << "\n";
}

uint32_t DataHandler::convertToLittleEndian(const unsigned char *bytes)
{
    return (uint32_t)(bytes[0] << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3]);
}
