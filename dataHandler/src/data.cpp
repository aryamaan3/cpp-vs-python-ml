#include "data.hpp"

data::data()
{
    _featureVector = std::make_shared<std::vector<uint8_t>>();
}

data::~data()
{
}

void data::setFeatureVector(std::shared_ptr<std::vector<uint8_t>> aFeatureVector)
{
    _featureVector = aFeatureVector;
}

void data::appendFeatureVector(uint8_t aFeature)
{
    _featureVector->push_back(aFeature);
}

void data::setLabel(uint8_t aLabel)
{
    _label = aLabel;
}

void data::setEnumLabel(int aEnumLabel)
{
    _enumLabel = aEnumLabel;
}

int data::getFeatureVectorSize()
{
    return _featureVector->size();
}

std::shared_ptr<std::vector<uint8_t>> data::getFeatureVector()
{
    return _featureVector;
}

uint8_t data::getLabel()
{
    return _label;
}

int data::getEnumLabel()
{
    return _enumLabel;
}