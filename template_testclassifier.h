#ifndef TEMPLATE_TESTCLASSIFIER_H
#define TEMPLATE_TESTCLASSIFIER_H

#include <vector>
#include <iostream>
#include <cmath>

#include <fstream>
#include <sstream>
#include <tbb/tbb.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/unordered_map.hpp>
#include <boost/filesystem.hpp>

#include "common.h"
#include "stronghypothesis.h"
#include "scanner.h"
#include "weakhypothesis.h"
#include "sampleextractor.h"



/*
The algorithms implemented in this program are based on "An introduction to ROC analysis"
from Tom Fawcett, 2005, Elsevier. If things seem confusing, it is recommended that you
read that paper.
*/



/**
 * When iterating over the images, a RocScanner will produce instances of this class.
 */
struct ScannerEntry
{
    cv::Rect position;
    float featureValue;
    bool isTrueFaceRegion;

    ScannerEntry() : position(0, 0, 0, 0), featureValue(0), isTrueFaceRegion(false) {}

    ScannerEntry(const cv::Rect & position_,
                 const float featureValue_,
                 const bool validDetection_) : position(position_),
                                               featureValue(featureValue_),
                                               isTrueFaceRegion(validDetection_) {}

    /**
     * The "natural" ordering for a list of instances of this class is the decreasing order.
     */
    bool operator < (const ScannerEntry & rh) const
    {
        return featureValue > rh.featureValue;
    }
};



inline cv::Point2f center(const cv::Rect & rect)
{
    return cv::Point(rect.x + rect.width/2.0f, rect.y + rect.height/2.0f);
}



/**
 * The criteria implemented here was taken from Pavani's article "Haar-like features with optimally
 * weighted rectangles for rapid object detection". If rectangle r matches a ground truth considering
 * Pavani's matching criteria, this function returns true.
 */
inline bool matchesGroundTruth(const cv::Rect & r, const std::vector<cv::Rect> & groundTruths)
{
    for (std::vector<cv::Rect>::const_iterator groundTruth = groundTruths.begin(); groundTruth != groundTruths.end(); ++groundTruth )
    {
        if ( groundTruth->width  * 0.9f <=  r.width  && r.width  <= groundTruth->width  * 1.1f
          && groundTruth->height * 0.9f <=  r.height && r.height <= groundTruth->height * 1.1f )
        {
            const cv::Point2f rCenter  = center(r);
            const cv::Point2f gtCenter = center(*groundTruth);
            float distance = cv::norm(gtCenter - rCenter);

            if ( distance <= groundTruth->width * 0.1f && distance <= groundTruth->height * 0.1f )
            {
                return true;
            }
        }
    }

    return false;
}



/**
 * Iterates over an image producing instances of ScannerEntry. Latter, such entries will be processed
 * so a ROC curve is produced.
 */
template<typename WeakClassifierType>
class RocScanner
{
public:
    RocScanner(StrongHypothesis<WeakClassifierType> * const classifier_) : classifier(classifier_),
                                                                           initial_size(20),
                                                                           scaling_factor(1.25f),
                                                                           delta(1.5f) {}

    void scan(const cv::Mat & image, const std::vector<cv::Rect> & groundTruth,
              tbb::concurrent_vector<ScannerEntry> & entries,
              unsigned int & positiveInstances,
              unsigned int & negativeInstances)
    {
        cv::Mat integralSum   (image.rows + 1, image.cols + 1, cv::DataType<double>::type);
        cv::Mat integralSquare(image.rows + 1, image.cols + 1, cv::DataType<double>::type);
        cv::integral(image, integralSum, integralSquare, cv::DataType<double>::type);

        //TODO review how we iterate over the image. This ROI seems to be iterating over the original
        //image, but it is also transformed in the ROI that iterates over the integrals. It is confusing
        //and probably performs worse then it could be.
        cv::Rect roi(0, 0, initial_size, initial_size);

        for(float scale = 1.5f; scale * initial_size <= image.cols
                             && scale * initial_size <= image.rows; scale *= scaling_factor)
        {
            const float shift = delta * scale; //like Viola and Jones do

            roi.width = roi.height = initial_size * scale;

            for (roi.x = 1; roi.x <= integralSum.cols - roi.width; roi.x += shift)
            {
                for (roi.y = 1; roi.y <= integralSum.rows - roi.height; roi.y += shift)
                {
                    cv::Rect exampleRoi = roi;
                    --exampleRoi.x; //Correctly position the roi over the integral images
                    --exampleRoi.y; //Note that we iterate over x and y from 1 (see the fors above)
                    exampleRoi.width = exampleRoi.height = exampleRoi.height + 1; //integral images have +1 sizes if compared to the original image

                    const Example example(integralSum(exampleRoi), integralSquare(exampleRoi));
                    const bool isFaceRegion = matchesGroundTruth(exampleRoi, groundTruth);

                    ScannerEntry e( exampleRoi, classifier->classificationValue(example, scale), isFaceRegion);
                    entries.push_back(e);

                    positiveInstances += isFaceRegion;
                    negativeInstances += !isFaceRegion;
                }
            }
        }
    }

private:
    StrongHypothesis<WeakClassifierType> const * const classifier;
    const int initial_size; //initial width and height of the detector
    const float scaling_factor; //how mutch the scale will change per iteration
    const float delta; //window shift constant
};



struct ImageAndGroundTruth
{
    cv::Mat image;
    std::vector<cv::Rect> faces;
};



/**
 * Uses the RocScanner to scan many images in parallel.
 */
template<typename WeakHypothesisType>
struct ParallelScan
{
    std::vector<ImageAndGroundTruth> * const images;
    unsigned int * const totalPositiveInstances;
    unsigned int * const totalNegativeInstances;
    unsigned int * const evaluatedImages;
    StrongHypothesis<WeakHypothesisType> * const strongHypothesis;
    tbb::concurrent_vector<ScannerEntry> * const entries;
    tbb::queuing_mutex * const mutex;

    ParallelScan(std::vector<ImageAndGroundTruth> * const images_,
                 unsigned int * const totalPositiveInstances_,
                 unsigned int * const totalNegativeInstances_,
                 unsigned int * const evaluatedImages_,
                 StrongHypothesis<WeakHypothesisType> * const strongHypothesis_,
                 tbb::concurrent_vector<ScannerEntry> * const entries_,
                 tbb::queuing_mutex * const mutex_) : images(images_),
                                                      totalPositiveInstances(totalPositiveInstances_),
                                                      totalNegativeInstances(totalNegativeInstances_),
                                                      evaluatedImages(evaluatedImages_),
                                                      strongHypothesis(strongHypothesis_),
                                                      entries(entries_),
                                                      mutex(mutex_) {}

    void operator()(tbb::blocked_range< unsigned int > & range) const
    {
        unsigned int positiveInstancesCount = 0;
        unsigned int negativeInstancesCount = 0;
        RocScanner<WeakHypothesisType> scanner(strongHypothesis);

        for(unsigned int k = range.begin(); k != range.end(); ++k)
        {
            ImageAndGroundTruth imageAndGt = (*images)[k];
            scanner.scan(imageAndGt.image, imageAndGt.faces, *entries, positiveInstancesCount, negativeInstancesCount);
        }

        {
            tbb::queuing_mutex::scoped_lock lock(*mutex);
            *totalPositiveInstances += positiveInstancesCount;
            *totalNegativeInstances += negativeInstancesCount;
            *evaluatedImages += range.size();

            std::cout << "\rProgress " << 100 * (*evaluatedImages) / images->size() << '%';
            std::cout.flush();

            lock.release();
        }
    }
};



/**
 * A point in a ROC curve.
 */
struct RocPoint
{
    unsigned int falsePositives;
    unsigned int truePositives;

    RocPoint() : falsePositives(0),
                 truePositives(0) {}

    bool operator < (const RocPoint & rh) const
    {
        return falsePositives < rh.falsePositives;
    }
};



inline double trapezoid_area(const double x1, const double x2, const double y1, const double y2)
{
    //As seen in "An introduction to ROC analysis, from Tom Fawcett, 2005, Elsevier."
    return std::abs(x1 - x2) * ((y1 + y2) / 2.0f);
}

void scannerEntries2RocCurve(const unsigned int total_positives,
                             const unsigned int total_negatives,
                             tbb::concurrent_vector<ScannerEntry> & entries,
                             std::vector<RocPoint> & rocCurve,
                             double & area_under_curve)
{
    //As seen in "An introduction to ROC analysis, from Tom Fawcett, 2005, Elsevier."
    tbb::parallel_sort(entries.begin(), entries.end());

    unsigned int false_positives = 0;
    unsigned int true_positives = 0;
    unsigned int false_positives_prev = 0;
    unsigned int true_positives_prev = 0;

    area_under_curve = .0f;

    float f_prev = -std::numeric_limits<float>::max(); //http://stackoverflow.com/questions/3529394/obtain-minimum-negative-float-value-in-c

    for(tbb::concurrent_vector<ScannerEntry>::iterator entry = entries.begin(); entry != entries.end(); ++entry)
    {
        if ( entry->featureValue != f_prev )
        {
            RocPoint p;
            p.truePositives = true_positives;
            p.falsePositives = false_positives;
            rocCurve.push_back(p);

            area_under_curve += trapezoid_area(false_positives, false_positives_prev, true_positives, true_positives_prev);
            false_positives_prev = false_positives;
            true_positives_prev = true_positives;

            f_prev = entry->featureValue;
        }

        true_positives  +=  entry->isTrueFaceRegion;
        false_positives += !entry->isTrueFaceRegion;
    }

    RocPoint p;
    p.truePositives = true_positives;
    p.falsePositives = false_positives;
    rocCurve.push_back(p); //This is 1, 1

    area_under_curve += trapezoid_area(total_negatives, false_positives_prev, total_negatives, true_positives_prev);
    area_under_curve /= (double)total_positives * total_negatives; // scale from P * N onto the unit square
}



typedef boost::unordered_map< std::string, std::vector<cv::Rect> > GroundTruthMap;

/**
 * Loads the instance of GroundTruthMap gtmap with ground truth data found in groundTruthFile.
 * The total amount of faces will be recorded in totalFaces output parameter.
 */
bool getGroundTruth(const std::string groundTruthFile, GroundTruthMap & gtmap, int & totalFaces)
{
    std::ifstream truthStream(groundTruthFile.c_str());
    if (!truthStream.is_open())
    {
        return false;
    }

    while( !truthStream.eof() )
    {
        std::string line;
        std::getline(truthStream, line);

        if (line.empty())
        {
            break;
        }

        std::string imageFileName;
        cv::Point2f leftEye, rightEye;

        std::istringstream lineStream(line);
        lineStream >> imageFileName
                   >> leftEye.x
                   >> leftEye.y
                   >> rightEye.x
                   >> rightEye.y;

        //Calculate face region from eye position.
        //Here this is done exactly as I extract faces from the BioId database.
        const float distanceBetweenEyes = cv::norm(rightEye-leftEye);
        const float roiWidthHeight = distanceBetweenEyes / 0.5154f;
        cv::Rect faceRegion(rightEye.x - roiWidthHeight * 0.2423f,
                            rightEye.y - roiWidthHeight * 0.25f,
                            roiWidthHeight, roiWidthHeight);

        if ( gtmap.find(imageFileName) == gtmap.end() )
        {
            std::vector<cv::Rect> rects;
            rects.push_back(faceRegion);

            gtmap.insert( std::make_pair(imageFileName, rects) );
        }
        else
        {
            gtmap[imageFileName].push_back(faceRegion);
        }

        ++totalFaces;
    }

    return true;
}



bool getTestImages(const std::string indexFileName,
                   const std::string groundTruthFileName,
                   std::vector<ImageAndGroundTruth> & images,
                   int & totalFacesInGroundTruth)
{
    totalFacesInGroundTruth = 0;
    GroundTruthMap gtmap;

    if( !getGroundTruth(groundTruthFileName, gtmap, totalFacesInGroundTruth) )
    {
        return false;
    }

    std::ifstream indexStream(indexFileName.c_str());
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

        ImageAndGroundTruth iagt;
        iagt.image = cv::imread(imagePath, cv::DataType<unsigned char>::type);
        if ( !iagt.image.data )
        {
            return false;
        }
        const std::string filename = boost::filesystem::path(imagePath).filename().native();
        if (gtmap.find(filename) == gtmap.end())
        {
            iagt.faces = std::vector<cv::Rect>(0);
        }
        else
        {
            iagt.faces = gtmap.at(filename);
        }

        images.push_back(iagt);
    }

    indexStream.close();
    return true;
}



/**
 * Use getTestImages__2 to test the classifier with an image database like those used in trainning,
 * i.e., a huge image composed of many 20x20 pixels images.
 */
bool getTestImages__2(const std::string positivesFile,
                      const std::string negativesFile,
                      std::vector<ImageAndGroundTruth> & images,
                      int & totalFacesInGroundTruth)
{
    std::vector<cv::Mat> samples;

    if ( !SampleExtractor::fromImageFile(positivesFile, samples) )
    {
        return false;
    }

    totalFacesInGroundTruth = samples.size();
    for(std::vector<cv::Mat>::iterator sample = samples.begin(); sample != samples.end(); ++sample)
    {
        ImageAndGroundTruth iagt;
        iagt.image = *sample;
        iagt.faces.push_back(cv::Rect(0, 0, sample->cols, sample->rows));

        images.push_back(iagt);
    }

    samples.clear();

    if ( !SampleExtractor::fromImageFile(negativesFile, samples) )
    {
        return false;
    }
    for(std::vector<cv::Mat>::iterator sample = samples.begin(); sample != samples.end(); ++sample)
    {
        ImageAndGroundTruth iagt;
        iagt.image = *sample;

        images.push_back(iagt);
    }

    return true;
}



template<typename WeakHypothesisType>
int ___main(const std::string testImagesIndexFileName,
            const std::string groundTruthFileName,
            const std::string strongHypothesisFile,
            const std::string rocCurveFile)
{
    StrongHypothesis<WeakHypothesisType> strongHypothesis;
    {
        std::ifstream in(strongHypothesisFile.c_str());
        if ( !in.is_open() )
        {
            return 7;
        }
        if ( !strongHypothesis.read(in) )
        {
            return 11;
        }

        std::cout << "Loaded strong classifier." << std::endl;
    }



    int totalFacesInGroundTruth = 0;
    std::vector<ImageAndGroundTruth> images;

    //I can use the getTestImages__2 to test the classifier an image database like those used in trainning, i.e.,
    //a huge image composed of many 20x20 pixels images.
    //If doinf this, remember to also change the name of the ___main function parameters.
    //if ( !getTestImages__2(positivesFile, negativesFile, images, totalFacesInGroundTruth) )
    if ( !getTestImages(testImagesIndexFileName, groundTruthFileName, images, totalFacesInGroundTruth) )
    {
        return 13;
    }
    std::cout << "Loaded " << images.size() << " images and " << totalFacesInGroundTruth << " ground truth entries." << std::endl;



    unsigned int totalPositiveInstances = 0;
    unsigned int totalNegativeInstances = 0;
    tbb::concurrent_vector<ScannerEntry> entries;
    {
        unsigned int evaluatedImages = 0;

        std::cout << "\rProgress 0%";
        std::cout.flush();

        tbb::queuing_mutex mutex;
        tbb::parallel_for(tbb::blocked_range< unsigned int >(0, images.size()),
                          ParallelScan<WeakHypothesisType>(&images,
                                                           &totalPositiveInstances,
                                                           &totalNegativeInstances,
                                                           &evaluatedImages,
                                                           &strongHypothesis,
                                                           &entries,
                                                           &mutex) );

        std::cout << "\rTotal positive windows: " << totalPositiveInstances;
        std::cout << "\nTotal negative windows: " << totalNegativeInstances;
        std::cout << "\nTotal scanned windows : " << totalPositiveInstances + totalNegativeInstances << std::endl;
    }

    std::cout << "\nBuilding ROC curve..." << std::endl;
    double areaUnderTheCurve = .0f;
    std::vector<RocPoint> rocCurve;
    scannerEntries2RocCurve(totalPositiveInstances, totalNegativeInstances, entries, rocCurve, areaUnderTheCurve);
    std::cout << "\rBuilt a ROC curve with " << rocCurve.size() << " ROC points and total area " << areaUnderTheCurve << ".\n";

    {
        std::cout << "\nWriting ROC curve to file " << rocCurveFile << '.' << std::endl;

        std::ofstream rocOut(rocCurveFile.c_str());
        if ( !rocOut.is_open() )
        {
            return 13;
        }
        for (std::vector<RocPoint>::iterator rocPoint = rocCurve.begin(); rocPoint != rocCurve.end(); ++rocPoint)
        {
            rocOut << rocPoint->falsePositives << ' ' << rocPoint->truePositives << std::endl;
            rocOut.flush();
        }

        rocOut.close();
    }

    return 0;
}

#endif // TEMPLATE_TESTCLASSIFIER_H
