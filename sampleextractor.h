#ifndef SAMPLEEXTRACTOR_H
#define SAMPLEEXTRACTOR_H

#include <string>

#include "common.h"
#include "labeledexample.h"



class SampleExtractor
{
private:
    SampleExtractor();

public:
    static bool extractRandomSample(const unsigned int sample_size,
                                    const std::string & filename,
                                    std::vector<LabeledExample> & samples,
                                    Classification c,
                                    std::vector<unsigned int> *sampleIndexes = 0);

    static bool fromIndexFile(const std::string &indexPath, std::vector<LabeledExample> &samples, Classification c);

    static bool fromImageFile(const std::string &imagePath, std::vector<LabeledExample> &samples, Classification c);
};

#endif // SAMPLEEXTRACTOR_H
