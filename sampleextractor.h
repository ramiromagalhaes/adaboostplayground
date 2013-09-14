#ifndef SAMPLEEXTRACTOR_H
#define SAMPLEEXTRACTOR_H

#include <string>
#include <opencv2/core/core.hpp>

#include "common.h"
#include "labeledexample.h"



class SampleExtractor
{
private:
    SampleExtractor();

public:
    static bool extractRandomSample(const unsigned int sample_size, const std::string & filename, std::vector<LabeledExample> & samples, Classification c);

    static bool fromIndexFile(const std::string &indexPath, std::vector<LabeledExample> &samples, Classification c);
};

#endif // SAMPLEEXTRACTOR_H
