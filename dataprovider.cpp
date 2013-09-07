#include "dataprovider.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

DataProvider::DataProvider(boost::filesystem::path & pathIndexPositives,
                           boost::filesystem::path & pathIndexNegatives,
                           unsigned int maxObjectsInBuffer_)
{
    initBuffers(maxObjectsInBuffer_);

    streamPositives.open(pathIndexPositives);
    streamNegatives.open(pathIndexNegatives);

    if ( !streamPositives.is_open() || !streamNegatives.is_open() )
    {
        throw 100;
    }

    //assuming the positive set can fit in memory, we load and store them
    load(streamPositives, 0, 0, positives, yes);
    streamPositives.close();

    //then we count how many images we have
    totalPositives = positives.size();

    //we count the negatives seeking for newlines.
    totalNegatives = std::count(std::istreambuf_iterator<char>(streamNegatives),
                                std::istreambuf_iterator<char>(), '\n');
    streamNegatives.seekg(0);
}


DataProvider::~DataProvider()
{
    positives.clear();
    samples.clear();
    streamPositives.close();
    streamNegatives.close();
}


bool DataProvider::loadNext()
{
    unsigned int totalLoaded = 0;
    unsigned int amountToLoad = maxObjectsInBuffer;

    if (currentLoad == 0) //positives not yet loaded to user. They will always come first.
    {
        for(unsigned int i = 0; i < positives.size(); ++i)
        {
            pushIntoSample(i, positives[i]);
        }

        totalLoaded = totalPositives;
        amountToLoad -= totalPositives;
    }

    totalLoaded += load(streamNegatives, totalPositives, amountToLoad, samples, no);

    if (!totalLoaded)
    {
        return false;
    }

    if (totalLoaded != maxObjectsInBuffer)
    {
        samples.resize(totalLoaded);
    }

    ++currentLoad;

    return true;
}

int DataProvider::load(fs::ifstream & stream,
                       const unsigned int offset,
                       const unsigned int amount,
                       std::vector< LabeledExample > & target,
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

        if ( amount > 0 ? i >= amount : false )
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
    streamNegatives.seekg(0);
    currentLoad = 0;

    samples.reserve(maxObjectsInBuffer);
}

std::vector<LabeledExample>::size_type DataProvider::size()
{
    return totalPositives + totalNegatives;
}

std::vector<LabeledExample>::size_type DataProvider::sizePositives()
{
    return totalPositives;
}

std::vector<LabeledExample>::size_type DataProvider::sizeNegatives()
{
    return totalNegatives;
}

std::vector<LabeledExample> const * const DataProvider::getCurrentBuffer()
{
    return &samples;
}

void DataProvider::initBuffers(int maxBuffer)
{
    maxObjectsInBuffer = maxBuffer;
    samples.reserve(maxObjectsInBuffer);

    currentLoad = 0;
}

void DataProvider::pushIntoSample(std::vector< LabeledExample >::size_type index, const LabeledExample &s)
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
