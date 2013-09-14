#ifndef SAMPLEEXTRACTOR_H
#define SAMPLEEXTRACTOR_H

#include <string>
#include <opencv2/core/core.hpp>

#include "common.h"



class SampleExtractor
{
private:
    SampleExtractor();

public:
    static bool extractRandomSample(const unsigned int sample_size, const std::string & filename, std::vector<cv::Mat> & samples);

    static bool fromIndexFile(const std::string &indexPath, std::vector<cv::Mat> &samples);
};

#endif // SAMPLEEXTRACTOR_H
