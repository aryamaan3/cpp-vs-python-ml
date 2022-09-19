#pragma once

#include <string>
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include <memory>

class data
{
    std::shared_ptr<std::vector<uint8_t>> _featureVector;
    uint8_t _label;
    int _enumLabel; // to compare label as int, A->1 B->2 C->3...

public:
    data();
    ~data();

    void setFeatureVector(std::shared_ptr<std::vector<uint8_t>> featureVector);
    void appendFeatureVector(uint8_t feature);
    void setLabel(uint8_t label);
    void setEnumLabel(int enumLabel);

    int getFeatureVectorSize();
    std::shared_ptr<std::vector<uint8_t>> getFeatureVector();
    uint8_t getLabel();
    int getEnumLabel();
};