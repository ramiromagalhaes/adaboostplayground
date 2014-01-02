#ifndef TESTDATABASE_H
#define TESTDATABASE_H

#include <string>
#include <vector>

#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/unordered_map.hpp>
#include <boost/filesystem.hpp>



struct ImageAndGroundTruth
{
    cv::Mat image;
    std::vector<cv::Rect> faces;
};



typedef boost::unordered_map< std::string, ImageAndGroundTruth > ImageAndGroundTruthMap;



class TestDatabase
{
public:
    TestDatabase();

    bool load(const std::string & imageIndexPath, const std::string & groundTruthPath);

    std::vector<ImageAndGroundTruth> getImagesAndGroundTruthAsVector() const;

    /**
     * Returns the amount of annotated faces in the images.
     */
    int size_annotations() const;

    /**
     * Returns the amount of images in the database.
     */
    int size_images() const;

private:
    bool loadImages(const std::string & imageIndexPath);
    bool loadGroundTruth(const std::string & grountTruthPath);

    ImageAndGroundTruthMap images;
    int totalFaces;
};

#endif // TESTDATABASE_H
