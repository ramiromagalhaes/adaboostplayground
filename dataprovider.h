#ifndef DATAPROVIDER_H
#define DATAPROVIDER_H

#include "Common.h"
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

namespace fs = boost::filesystem;


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

    std::vector<LabeledExample> const * const getCurrentBuffer();

    void reset();

    /**
     * @brief size Returns the total amount of samples in this collection.
     */
    std::vector<LabeledExample>::size_type size();
    /**
     * @brief size Returns the size of the positive samples set.
     */
    std::vector<LabeledExample>::size_type sizePositives();
    /**
     * @brief size Returns the size of the negative samples set.
     */
    std::vector<LabeledExample>::size_type sizeNegatives();


private:
    int load(boost::filesystem::ifstream &stream, const unsigned int offset, const unsigned int amount, std::vector< LabeledExample > & target, const Classification classification);
    void initBuffers(int maxBuffer);
    void pushIntoSample(std::vector<LabeledExample>::size_type index, const LabeledExample &s);

    fs::ifstream streamPositives,
                 streamNegatives;
    unsigned int totalPositives,
                 totalNegatives;

    unsigned int maxObjectsInBuffer;
    unsigned int currentLoad;

    std::vector< LabeledExample > positives;
    std::vector< LabeledExample > samples;
};

#endif // DATAPROVIDER_H
