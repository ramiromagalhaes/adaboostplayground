#ifndef DATAPROVIDER_H
#define DATAPROVIDER_H

#include <opencv2/core/core.hpp>
#include <boost/filesystem/fstream.hpp>

namespace fs = boost::filesystem;


class DataProvider
{
public:
    DataProvider();
    DataProvider(fs::path & indexPath_, int maxObjectsInBuffer_);
    ~DataProvider();

    bool loadNext();

    std::vector<cv::Mat> const * const getCurrentBuffer();

private:
    void initBuffers(int maxBuffer);
    bool prepareIndex();

    fs::path indexPath;
    fs::ifstream indexFile;
    bool triedToOpen;


    int maxObjectsInBuffer;
    int currentBatch;

    std::vector<cv::Mat> images;
};

#endif // DATAPROVIDER_H
