#include "data.hpp"

Data::Data()
{
    _featureVector = new std::vector<uint8_t>();
}

Data::~Data()
{
}

void Data::setFeatureVector(std::vector<uint8_t>* aFeatureVector)
{
    _featureVector = aFeatureVector;
}

void Data::appendFeatureVector(uint8_t aFeature)
{
    _featureVector->push_back(aFeature);
}

void Data::setLabel(uint8_t aLabel)
{
    _label = aLabel;
}

void Data::setEnumLabel(int aEnumLabel)
{
    _enumLabel = aEnumLabel;
}

void Data::setDistance(double aDistance)
{
    _distance = aDistance;
}

int Data::getFeatureVectorSize()
{
    return _featureVector->size();
}

std::vector<uint8_t>* Data::getFeatureVector()
{
    return _featureVector;
}

uint8_t Data::getLabel()
{
    return _label;
}

int Data::getEnumLabel()
{
    return _enumLabel;
}

double Data::getDistance()
{
    return _distance;
}
