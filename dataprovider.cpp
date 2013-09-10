#include "dataprovider.h"

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



DataProvider::DataProvider(const std::string & pathIndexPositives,
                           const std::string & pathIndexNegatives) : cache(oiio::ImageCache::create()),
                                                                     totalPositives(0),
                                                                     totalNegatives(0)
{
    cache->attribute ("max_memory_MB", 40000.0); //40GB default

    if ( !parseIndexFile(pathIndexPositives, positiveFiles, cache, totalPositives)
         & !parseIndexFile(pathIndexNegatives, negativeFiles, cache, totalNegatives) ) //notice: this single & is intentional
    {
        //TODO Throw a nice exception
        throw 200;
    }

    reset();
}


DataProvider::~DataProvider()
{
    oiio::ImageCache::destroy(cache);
}


void DataProvider::reset()
{
    currentSource = &positiveFiles;
    currentImage = 0;
    currentX = 0;
}


bool DataProvider::nextSample(LabeledExample & sample) //TODO assert the Mat sizes
{
    oiio::ImageSpec currentSpec;
    oiio::ustring currentImageName( currentSource->at(currentImage) ); //TODO check "at" function
    if( !cache->get_imagespec(currentImageName, currentSpec) )
    {
        throw 20;
    }

    if (currentX >= currentSpec.width) //TODO check if 20 divides spec.width somewhere
    {
        if ( currentImage >= currentSource->size() - 1)
        {
            if ( currentSource != &positiveFiles )
            {
                return false;
            }

            currentSource = &negativeFiles;
            currentImage = 0;
            currentX = 0;
        }
        else
        {
            ++currentImage;
            currentX = 0;
        }
    }

    cv::Mat image(20, 20, CV_8UC1);
    //TODO get_pixels returns boolean. probably should use that too.
    cache->get_pixels( currentImageName,          //the file
                       0, 0,                      //subimage and texture or whatever
                       currentX, currentX + 20,   //x
                       0, 20,                     //y
                       0, 1, 0, 1,                //z and channels
                       oiio::TypeDesc::UCHAR,           //the type of data found in the cv::Mat object
                       image.data );     //the pixels of the image are written into the matrix

    sample.updateIntegrals(image);
    sample.label = currentSource == &positiveFiles ? yes : no;

    currentX += SAMPLE_WIDTH;

    return true;
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



bool DataProvider::parseIndexFile(const std::string & indexPath,
                               std::vector<std::string> & imagesPaths,
                               oiio::ImageCache * const cache,
                               unsigned int & total)
{
    std::ifstream indexStream(indexPath.c_str());
    if (!indexStream.is_open())
    {
        return false;
    }

    oiio::ImageSpec spec;
    while( !indexStream.eof() )
    {
        std::string imagePathString;
        std::getline(indexStream, imagePathString);

        if (imagePathString.empty())
        {
            break;
        }

        imagesPaths.push_back(imagePathString);
        oiio::ImageBuf buffer(imagePathString, cache); //insert a buffer in a cache.

        cache->get_imagespec(oiio::ustring(imagePathString), spec); //weird way to read the specs, but at least it works
        total += spec.width / SAMPLE_WIDTH; //TODO assuming that a lot of things are alright, this should work
    }

    indexStream.close();

    return true;
}
