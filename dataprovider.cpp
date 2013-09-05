#include "dataprovider.h"

DataProvider::DataProvider()
{
    initBuffers(100000);
}

DataProvider::DataProvider(boost::filesystem::path &indexPath_, int maxObjectsInBuffer_)
{
    initBuffers(maxObjectsInBuffer_);

    indexPath = indexPath_;
    triedToOpen = false;
}

DataProvider::~DataProvider()
{
    images.clear();
    if ( indexFile.is_open() )
    {
        indexFile.close();
    }
}


bool DataProvider::loadNext()
{
    if ( !prepareIndex() )
    {
        return false;
    }

    ++currentBatch;

    for (int i = 0; i < maxObjectsInBuffer; ++i)
    {

    }
}

const std::vector<cv::Mat> * const DataProvider::getCurrentBuffer()
{
    return &images;
}

void DataProvider::initBuffers(int maxBuffer)
{
    maxObjectsInBuffer = maxBuffer;
    currentBatch = 0;
    images.reserve(maxObjectsInBuffer);
}

bool DataProvider::prepareIndex()
{
    if ( !triedToOpen )
    {
        indexFile.open(indexPath, std::ifstream::in);
        triedToOpen = true;
    }

    return indexFile.is_open();
}
