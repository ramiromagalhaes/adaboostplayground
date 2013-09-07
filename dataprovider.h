#ifndef DATAPROVIDER_H
#define DATAPROVIDER_H

#include "Common.h"
#include <boost/filesystem.hpp>
#include <fstream>


namespace fs = boost::filesystem;


//TODO if the data provider buffer is bigger than the amount of images, simply retain all of them in memory.
class DataProvider
{
public:
    DataProvider(fs::path & pathIndexPositives, fs::path & pathIndexNegatives, unsigned int maxObjectsInBuffer_);
    ~DataProvider();

    /**
     * @brief Load the next objects in the current buffer.
     * @return
     */
    bool loadNext();

    LEContainer const * const getCurrentBuffer();

    void reset();

    /**
     * @brief size Returns the total amount of samples in this collection.
     */
    LEContainer::size_type size();
    /**
     * @brief size Returns the size of the positive samples set.
     */
    LEContainer::size_type sizePositives();
    /**
     * @brief size Returns the size of the negative samples set.
     */
    LEContainer::size_type sizeNegatives();


private:
    unsigned int load(std::ifstream &stream,
             const unsigned int offset,
             const unsigned int amount,
             LEContainer & target,
             const Classification classification);

    void initBuffers(int maxBuffer);

    void pushIntoSample(LEContainer::size_type index, const LabeledExample &s);

    std::ifstream streamPositives,
                  streamNegatives;
    unsigned int totalPositives,
                 totalNegatives;

    unsigned int maxObjectsInBuffer;
    unsigned int nextIndexToLoad; //all loading into the buffer should be done from this index on

    LEContainer positives;
    LEContainer samples;
};

#endif // DATAPROVIDER_H
