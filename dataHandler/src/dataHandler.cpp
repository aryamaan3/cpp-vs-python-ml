#include "dataHandler.hpp"
#include <iostream>

constexpr static auto OFFSET_FEATURES = 4;
constexpr static auto OFFSET_LABELS = 2;

dataHandler::dataHandler()
{
    _dataArray = std::make_shared<std::vector<std::shared_ptr<data>>>();
    _trainingData = std::make_shared<std::vector<std::shared_ptr<data>>>();
    _testData = std::make_shared<std::vector<std::shared_ptr<data>>>();
    _valData = std::make_shared<std::vector<std::shared_ptr<data>>>();
}

dataHandler::~dataHandler()
{
    //might be useless
}

void dataHandler::readFeatureVector(std::string path)
{
    uint32_t header[OFFSET_FEATURES]; // MAGIC, NBITEMS, NBROWS, NBCOLS
    unsigned char bytes[OFFSET_FEATURES];
    FILE *file = fopen(path.c_str(), "r");

    if (file == NULL)
    {
        std::cout << "Could not open file " << path << "\n";
        exit(1);
    }

    for (int i = 0; i < OFFSET_FEATURES; ++i)
    {
        fread(bytes, sizeof(bytes), 1, file);
        header[i] = convertToLittleEndian(bytes);
    }

    int itemSize = header[2] * header[3];
    for (int i = 0; i < header[1]; ++i)
    {
        std::shared_ptr<data> d = std::make_shared<data>();
        uint8_t element[1];
        for (int j = 0; j < itemSize; ++j)
        {
            fread(element, sizeof(element), 1, file);
            d->appendFeatureVector(element[0]);
        }
        _dataArray->push_back(d);
    }

    std::cout << "Retrieved features, size = " << _dataArray->size() << "\n";
}

void dataHandler::readLabel(std::string path)
{
    uint32_t header[OFFSET_LABELS]; // MAGIC, NBITEMS
    unsigned char bytes[OFFSET_FEATURES];
    FILE *file = fopen(path.c_str(), "rb");

    if (file == NULL)
    {
        std::cout << "Could not open file " << path << "\n";
        exit(1);
    }

    for (int i = 0; i < OFFSET_LABELS; ++i)
    {
        fread(bytes, sizeof(bytes), 1, file);
        header[i] = convertToLittleEndian(bytes);
    }

    std::cout << "Retrieved label header from " << path << "\n";
    std::cout << "Magic number: " << header[0] << "\n";
    std::cout << "Number of items = " << header[1] << "\n";

    for (int i = 0; i < header[1]; ++i)
    {
        uint8_t element[1];
        fread(element, sizeof(element), 1, file);
        _dataArray->at(i)->setLabel(element[0]);
    }

    std::cout << "Retrieved labels, size = " << _dataArray->size() << "\n";
}

void dataHandler::splitData()
{
    std::unordered_set<int> usedIndexes;
    int trainingDataSize = _dataArray->size() * TRAINING_DATA_PERCENT;
    int testDataSize = _dataArray->size() * TEST_DATA_PERCENT;
    int valDataSize = _dataArray->size() * VAL_DATA_PERCENT;

    for (auto i = 0; i < trainingDataSize; ++i)
    {
        auto index = rand() % _dataArray->size();
        while (usedIndexes.find(index) != usedIndexes.end())
        {
            index = rand() % _dataArray->size();
        }
        usedIndexes.insert(index);
        _trainingData->push_back(_dataArray->at(index));
    }

    for (auto i = 0; i < testDataSize; ++i)
    {
        auto index = rand() % _dataArray->size();
        while (usedIndexes.find(index) != usedIndexes.end())
        {
            index = rand() % _dataArray->size();
        }
        usedIndexes.insert(index);
        _testData->push_back(_dataArray->at(index));
    }

    for (auto i = 0; i < valDataSize; ++i)
    {
        auto index = rand() % _dataArray->size();
        while (usedIndexes.find(index) != usedIndexes.end())
        {
            index = rand() % _dataArray->size();
        }
        usedIndexes.insert(index);
        _valData->push_back(_dataArray->at(index));
    }

    std::cout << "Training data size = " << _trainingData->size() << "\n";
    std::cout << "Test data size = " << _testData->size() << "\n";
    std::cout << "Val data size = " << _valData->size() << "\n";
}


void dataHandler::countClasses()
{
    int count = 0;
    for (auto i = 0U; i < _dataArray->size(); ++i)
    {
        if (_labelMap.find(_dataArray->at(i)->getLabel()) == _labelMap.end())
        {
            _labelMap[_dataArray->at(i)->getLabel()] = count;
            _dataArray->at(i)->setEnumLabel(count);
            ++count;
        }
    }
    _numClasses = count;
    std::cout << "Number of classes = " << _numClasses << "\n";
}



uint32_t dataHandler::convertToLittleEndian(const unsigned char *bytes)
{
    return (uint32_t) (bytes[0] << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3]);
}



std::shared_ptr<std::vector<std::shared_ptr<data>>> dataHandler::getTrainingData()
{
    return _trainingData;
}


std::shared_ptr<std::vector<std::shared_ptr<data>>> dataHandler::getTestData()
{
    return _testData;
}


std::shared_ptr<std::vector<std::shared_ptr<data>>> dataHandler::getValData()
{
    return _valData;
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

    return 0;
}
