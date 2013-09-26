#include "sampleextractor.h"

#include <fstream>
#include <cmath>
#include <ctime>
#include <utility>
#include <opencv2/highgui/highgui.hpp>
#include <boost/unordered_set.hpp>


SampleExtractor::SampleExtractor()
{
}

//TODO ensure that the same sample won't be chosen again.
bool SampleExtractor::extractRandomSample(const unsigned int sample_size,
                                          const std::string &imagePath,
                                          std::vector<LabeledExample> &samples,
                                          Classification c,
                                          std::vector<unsigned int > *sampleIndexes)
{
    boost::unordered_set<unsigned int> selectedIndexes;

    std::srand(std::time(0));

    const cv::Size roiSize(20 ,20);
    const cv::Mat full_image = cv::imread(imagePath, cv::DataType<unsigned char>::type);


    for (unsigned int i = 0; i < sample_size; ++i)
    {
        unsigned int sampleX = 0;
        bool did_insert = false;
        do
        {
            sampleX = (full_image.cols/roiSize.width) * ((float)std::rand() / RAND_MAX);
            did_insert = selectedIndexes.insert(sampleX).second;
        } while ( !did_insert );

        if (sampleIndexes)
        {
            sampleIndexes->push_back(sampleX);
        }

        cv::Rect roi(sampleX * 20, 0, roiSize.width, roiSize.height);

        cv::Mat image = cv::Mat(full_image, roi);
        if ( !image.data )
        {
            return false;
        }

        samples.push_back(LabeledExample(image, c));
    }

    return true;
}

bool SampleExtractor::fromIndexFile(const std::string & indexPath, std::vector<LabeledExample> &samples, Classification c)
{
    std::ifstream indexStream(indexPath.c_str());
    if (!indexStream.is_open())
    {
        return false;
    }

    while( !indexStream.eof() )
    {
        std::string imagePath;
        std::getline(indexStream, imagePath);

        if (imagePath.empty())
        {
            break;
        }

        //TODO check if all image sizes comply with a parameter
        cv::Mat image = cv::imread(imagePath, cv::DataType<unsigned char>::type);
        if ( !image.data )
        {
            return false;
        }

        samples.push_back(LabeledExample(image, c));
    }

    indexStream.close();

    return true;
}

bool SampleExtractor::fromImageFile(const std::string &imagePath, std::vector<LabeledExample> &samples, Classification c)
{
    const cv::Size roiSize(20 ,20);
    const cv::Mat full_image = cv::imread(imagePath, cv::DataType<unsigned char>::type);

    unsigned int total_images = full_image.cols / roiSize.width;
    samples.resize(total_images);
    for (unsigned int i = 0; i < total_images; ++i)
    {
        cv::Rect roi(i * 20, 0, roiSize.width, roiSize.height);

        cv::Mat image = cv::Mat(full_image, roi);
        if ( !image.data )
        {
            return false;
        }

        samples[i] = LabeledExample(image, c);
    }

    return true;
}

bool SampleExtractor::fromImageFile(const std::string &imagePath, std::vector<cv::Mat> &samples)
{
    const cv::Size roiSize(20 ,20);
    const cv::Mat full_image = cv::imread(imagePath, cv::DataType<unsigned char>::type);

    unsigned int total_images = full_image.cols / roiSize.width;
    samples.resize(total_images);
    for (unsigned int i = 0; i < total_images; ++i)
    {
        cv::Rect roi(i * 20, 0, roiSize.width, roiSize.height);

        cv::Mat image = cv::Mat(full_image, roi);
        if ( !image.data )
        {
            return false;
        }

        samples[i] = image;
    }

    return true;
}
