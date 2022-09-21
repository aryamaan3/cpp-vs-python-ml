#pragma once

#include <string>
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include <memory>

class Data
{
    std::vector<uint8_t>* _featureVector;
    uint8_t _label;
    int _enumLabel; // to compare label as int, A->1 B->2 C->3...
    double _distance;

public:
    Data();
    ~Data();

    void setFeatureVector(std::vector<uint8_t>* featureVector);
    void appendFeatureVector(uint8_t feature);
    void setLabel(uint8_t label);
    void setEnumLabel(int enumLabel);
    void setDistance(double distance);

    int getFeatureVectorSize();
    std::vector<uint8_t>* getFeatureVector();
    uint8_t getLabel();
    int getEnumLabel();
    double getDistance();
};