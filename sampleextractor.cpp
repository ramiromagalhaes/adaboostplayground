#include "sampleextractor.h"

#include <fstream>
#include <cmath>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>

SampleExtractor::SampleExtractor()
{
}

//TODO ensure that the same sample won't be chosen again.
bool SampleExtractor::extractRandomSample(const unsigned int sample_size, const std::string &imagePath, std::vector<LabeledExample> &samples, Classification c)
{
    std::srand(std::time(0));

    const cv::Size roiSize(20 ,20);
    const cv::Mat full_image = cv::imread(imagePath, cv::DataType<unsigned char>::type);


    for (unsigned int i = 0; i < sample_size; ++i)
    {
        const unsigned int sampleX = (full_image.cols/roiSize.width) * ((float)std::rand() / RAND_MAX);

        cv::Rect roi(sampleX * 20, 0, roiSize.width, roiSize.height);

        cv::Mat image = cv::Mat(full_image, roi);
        if ( !image.data )
        {
            return false;
        }
        LabeledExample sample(image, c);
        samples.push_back(sample);
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

        LabeledExample sample(image, c);
        samples.push_back(sample);
    }

    indexStream.close();

    return true;
}
