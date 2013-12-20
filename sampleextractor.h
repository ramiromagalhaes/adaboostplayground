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

    /**
     * Randomly 'cuts' samples from a file.
     */
    static bool extractRandomSample(const unsigned int sample_size,
                                    const std::string & filename,
                                    std::vector<LabeledExample> & samples,
                                    Classification c,
                                    std::vector<unsigned int> *sampleIndexes = 0);

    /**
     * 'Cuts' samples from an image using an index. The index 'points' to parts of the big image.
     */
    static bool extractSamplesWithIndex(const std::string &imagePath, const std::string &indexPath, std::vector<LabeledExample> &samples, Classification c);

    /**
     * The index file contains the paths of images to be loaded.
     */
    static bool fromIndexFile(const std::string &indexPath, std::vector<LabeledExample> &samples, Classification c);

    static bool fromImageFile(const std::string &imagePath, std::vector<LabeledExample> &samples, Classification c);

    static bool fromImageFile(const std::string &imagePath, std::vector<cv::Mat> &samples);

};

#endif // SAMPLEEXTRACTOR_H
