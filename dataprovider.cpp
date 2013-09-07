#include "dataprovider.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

DataProvider::DataProvider(std::string & pathIndexPositives,
                           std::string & pathIndexNegatives,
                           unsigned int maxObjectsInBuffer_)
{
    initBuffers(maxObjectsInBuffer_);

    streamPositives.open(pathIndexPositives.c_str());
    streamNegatives.open(pathIndexNegatives.c_str());

    if ( !streamPositives.is_open() || !streamNegatives.is_open() )
    {
        throw 100;
    }

    //assuming the positive set can fit in memory, we load and store them
    load(streamPositives, 0, 0, positives, yes);
    streamPositives.close();

    //then we count how many images we have
    totalPositives = positives.size();

    //we count the negatives counting the newlines
    totalNegatives = std::count(std::istreambuf_iterator<char>(streamNegatives),
                                std::istreambuf_iterator<char>(), '\n');
    streamNegatives.clear();
    streamNegatives.seekg(0, std::ios::beg);
}


DataProvider::~DataProvider()
{
    positives.clear();
    samples.clear();
    streamPositives.close();
    streamNegatives.close();
}


/**
 * Loads a chunk of the data into the buffer. The amount loaded varies.
 */
bool DataProvider::loadNext()
{
    unsigned int positives_loaded = 0;
    if (nextIndexToLoad < positives.size()) //this assumes that the positive sizes are smaller than the buffer
    {
        samples.assign(positives.begin(), positives.end());
        positives_loaded = positives.size();
    }

    unsigned int totalLoaded = positives_loaded + load(streamNegatives, positives_loaded, maxObjectsInBuffer - positives_loaded, samples, no);
    nextIndexToLoad += totalLoaded;

    if (!totalLoaded)
    {
        return false;
    }

    if (totalLoaded != maxObjectsInBuffer)
    {
        samples.resize(totalLoaded);
    }

    return true;
}

unsigned int DataProvider::load(std::ifstream & stream,
                       const unsigned int offset,
                       const unsigned int amount,
                       LEContainer & target,
                       const Classification classification)
{
    unsigned int i = 0;
    for (;; ++i)
    {
        std::string imagePathString;
        std::getline(stream, imagePathString);

        if ( stream.eof() )
        {
            break;
        }

        if ( amount > 0 ? i > amount : false )
        {
            break;
        }

        cv::Mat img = cv::imread(imagePathString, cv::IMREAD_GRAYSCALE);
        if (img.data == 0)
        {
            std::cerr << "File " << imagePathString << " failed to load. Replacing with dummy.\n";
            img = cv::Mat(20, 20, CV_8U); //TODO fixed size dummy could be variable
        }

        //since I only reserved the memory, I need to decide between push_back or direct vector insertion
        if (offset + i >= target.size())
        {
            target.push_back(LabeledExample(img, classification));
        }
        else
        {
            //AFAIK, there is no need to worry about previously loaded images
            target[offset + i] = LabeledExample(img, classification);
        }
    }

    return i;
}

void DataProvider::reset()
{
    streamNegatives.clear();
    streamNegatives.seekg(0, std::ios::beg);
    nextIndexToLoad = 0;

    samples.reserve(maxObjectsInBuffer);
}

LEContainer::size_type DataProvider::size()
{
    return totalPositives + totalNegatives;
}

LEContainer::size_type DataProvider::sizePositives()
{
    return totalPositives;
}

LEContainer::size_type DataProvider::sizeNegatives()
{
    return totalNegatives;
}

LEContainer const * const DataProvider::getCurrentBuffer()
{
    return &samples;
}

void DataProvider::initBuffers(int maxBuffer)
{
    maxObjectsInBuffer = maxBuffer;
    samples.reserve(maxObjectsInBuffer);

    nextIndexToLoad = 0;
}

void DataProvider::pushIntoSample(LEContainer::size_type index, const LabeledExample &s)
{
    //since I only reserved the memory, I need to decide between push_back or direct vector insertion
    if (index >= samples.size())
    {
        samples.push_back(s);
    }
    else
    {
        //AFAIK, there is no need to worry about previously loaded images
        samples[index] = s;
    }
}
