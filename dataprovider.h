#ifndef DATAPROVIDER_H
#define DATAPROVIDER_H

#include "Common.h"
#include "LabeledExample.h"
#include <vector>
#include <string>
#include <OpenImageIO/imagebuf.h>



#define SAMPLE_WIDTH 20


namespace oiio = OIIO;



class DataProvider
{
public:
    DataProvider(const std::string & pathIndexPositives,
                 const std::string & pathIndexNegatives);
    ~DataProvider();

    void reset();

    /**
     * Returns the next sample.
     */
    bool nextSample(LabeledExample & sample);

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
    bool parseIndexFile(const std::string &indexPath, std::vector<std::string> &imagesPaths, oiio::ImageCache * const cache, unsigned int & total);

    std::vector<std::string> positiveFiles;
    std::vector<std::string> negativeFiles;

    oiio::ImageCache * const cache;

    unsigned int totalPositives,
                 totalNegatives;

    std::vector<std::string> * currentSource;
    unsigned int currentImage;
    unsigned int currentX;
};

#endif // DATAPROVIDER_H
