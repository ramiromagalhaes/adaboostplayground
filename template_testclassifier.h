#ifndef TEMPLATE_TESTCLASSIFIER_H
#define TEMPLATE_TESTCLASSIFIER_H

#include <vector>
#include <iostream>

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
#include "haarclassifier.h"
#include "sampleextractor.h"



struct ScannerEntry
{
    cv::Rect position;
    float featureValue;
    bool validDetection;

    ScannerEntry() : position(0, 0, 0, 0), featureValue(0), validDetection(false) {}

    ScannerEntry(const cv::Rect & position_,
                 const float featureValue_,
                 const bool validDetection_) : position(position_),
                                              featureValue(featureValue_),
                                              validDetection(validDetection_) {}

    bool operator < (const ScannerEntry & rh) const
    {
        return featureValue < rh.featureValue;
    }
};



cv::Point2f center(const cv::Rect & rect)
{
    return cv::Point(rect.x + rect.width/2.0f, rect.y + rect.height/2.0f);
}



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



template<typename WeakClassifierType>
class RocScanner
{
public:
    RocScanner(StrongHypothesis<WeakClassifierType> * const classifier_) : classifier(classifier_),
                                                                           initial_size(20),
                                                                           scaling_factor(1.25f),
                                                                           delta(1.5f) {}

    unsigned int scan(const cv::Mat & image, const std::vector<cv::Rect> & groundTruth, tbb::concurrent_vector<ScannerEntry> & entries)
    {
        cv::Mat integralSum   (image.rows + 1, image.cols + 1, cv::DataType<double>::type);
        cv::Mat integralSquare(image.rows + 1, image.cols + 1, cv::DataType<double>::type);
        cv::integral(image, integralSum, integralSquare, cv::DataType<double>::type);

        //const float max_scaling_factor = std::pow(scaling_factor, 9.0f);

        unsigned int scannedWindows = 0;

        //We iterate our ROI over the integral image even though it is set as the original image,
        //therefore, we must set the correct ROI when saving what we found here.
        cv::Rect roi(0, 0, initial_size, initial_size);

        for(float scale = 1.5f; scale * initial_size < image.cols
                             && scale * initial_size < image.rows
                             /*&& scale < max_scaling_factor*/; scale *= scaling_factor)
        {
            const float shift = delta * scale; //like Viola and Jones do

            roi.width = roi.height = initial_size * scale + 1; //integral images have +1 sizes if compared to the original image

            for (roi.x = 1; roi.x < integralSum.cols - roi.width; roi.x += shift)
            {
                for (roi.y = 1; roi.y < integralSum.rows - roi.height; roi.y += shift)
                {
                    cv::Rect exampleRoi = roi;
                    --exampleRoi.x; //Correctly position the roi over the integral images
                    --exampleRoi.y; //Note that we iterate over x and y from 1 (see the fors above)

                    const Example example(integralSum(exampleRoi), integralSquare(exampleRoi));
                    const bool isFaceRegion = matchesGroundTruth(exampleRoi, groundTruth);

                    ScannerEntry e( exampleRoi, classifier->classificationValue(example, scale), isFaceRegion);
                    entries.push_back(e);
                    ++scannedWindows;
                }
            }
        }

        return scannedWindows;
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



template<typename WeakHypothesisType>
struct ParallelScan
{
    std::vector<ImageAndGroundTruth> * const images;
    int * const totalScannedWindows;
    int * const evaluatedImages;
    StrongHypothesis<WeakHypothesisType> * const strongHypothesis;
    tbb::concurrent_vector<ScannerEntry> * const entries;
    tbb::queuing_mutex * const mutex;

    ParallelScan(std::vector<ImageAndGroundTruth> * const images_,
                 int * const totalScannedWindows_,
                 int * const evaluatedImages_,
                 StrongHypothesis<WeakHypothesisType> * const strongHypothesis_,
                 tbb::concurrent_vector<ScannerEntry> * const entries_,
                 tbb::queuing_mutex * const mutex_) : images(images_),
                                                      totalScannedWindows(totalScannedWindows_),
                                                      evaluatedImages(evaluatedImages_),
                                                      strongHypothesis(strongHypothesis_),
                                                      entries(entries_),
                                                      mutex(mutex_) {}

    void operator()(tbb::blocked_range< unsigned int > & range) const
    {
        unsigned int scannedWindows = 0;
        RocScanner<WeakHypothesisType> scanner(strongHypothesis);

        for(unsigned int k = range.begin(); k != range.end(); ++k)
        {
            ImageAndGroundTruth imageAndGt = (*images)[k];
            scannedWindows = scanner.scan(imageAndGt.image, imageAndGt.faces, *entries);
        }

        {
            tbb::queuing_mutex::scoped_lock lock(*mutex);
            *totalScannedWindows += scannedWindows;
            *evaluatedImages += 1;

            std::cout << "\rProgress " << 100 * (*evaluatedImages) / images->size() << '%';
            std::cout.flush();

            lock.release();
        }
    }
};



struct RocRecord
{
    int falsePositives;
    int truePositives;
    float value;

    RocRecord() : falsePositives(0), truePositives(0), value(0) {}

    bool operator < (const RocRecord & rh) const
    {
        return falsePositives < rh.falsePositives;
    }
};



template<typename WeakHypothesisType>
struct RocCalculator
{
    std::vector<LabeledExample> const * const samples;
    std::vector<RocRecord> * const records;
    StrongHypothesis<WeakHypothesisType> * const strongHypothesis;

    RocCalculator(std::vector<LabeledExample> const * const samples_,
                  std::vector<RocRecord> * records_,
                  StrongHypothesis<WeakHypothesisType> * const strongHypothesis_) : samples(samples_),
                                                                                    records(records_),
                                                                                    strongHypothesis(strongHypothesis_) {}

    void operator()(tbb::blocked_range< unsigned int > & range) const
    {
        for (unsigned int i = range.begin(); i < range.end(); ++i)
        {
            (*records)[i].value = strongHypothesis->classificationValue( (*samples)[i] );

            for (unsigned int j = 0; j < samples->size(); ++j)
            {
                const float classVal = strongHypothesis->classificationValue( (*samples)[j] );
                (*records)[i].truePositives  += (classVal >= (*records)[i].value) && ((*samples)[j].getLabel() == yes);
                (*records)[i].falsePositives += (classVal >= (*records)[i].value) && ((*samples)[j].getLabel() == no);
            }
        }
    }
};



typedef boost::unordered_map< std::string, std::vector<cv::Rect> > GroundTruthMap;



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



template<typename WeakHypothesisType>
struct TotalTruePositivesAndNegatives
{
    std::vector<ImageAndGroundTruth> * const images;
    int * const totalTruePositives;
    int * const totalFalsePositives;
    int * const totalScannedWindows;

    int * const evaluatedImages;

    StrongHypothesis<WeakHypothesisType> * const strongHypothesis;

    tbb::queuing_mutex * const mutex;

    TotalTruePositivesAndNegatives(std::vector<ImageAndGroundTruth> * const images_,
                                   int * const totalTruePositives_,
                                   int * const totalFalsePositives_,
                                   int * const totalScannedWindows_,
                                   int * const evaluatedImages_,
                                   tbb::queuing_mutex * const mutex_,
                                   StrongHypothesis<WeakHypothesisType> * const strongHypothesis_) : images(images_),
                                                                                                     totalTruePositives(totalTruePositives_),
                                                                                                     totalFalsePositives(totalFalsePositives_),
                                                                                                     totalScannedWindows(totalScannedWindows_),
                                                                                                     evaluatedImages(evaluatedImages_),
                                                                                                     strongHypothesis(strongHypothesis_),
                                                                                                     mutex(mutex_) {}

    void operator()(tbb::blocked_range< unsigned int > & range) const
    {
        unsigned int scannedWindows = 0;
        Scanner<WeakHypothesisType> scanner(strongHypothesis);

        for(unsigned int k = range.begin(); k != range.end(); ++k)
        {
            ImageAndGroundTruth imageAndGt = (*images)[k];

            int imageTruePositives = 0;

            std::vector<cv::Rect> detections;
            scannedWindows += scanner.scan(imageAndGt.image, detections);

            for(unsigned int i = 0; i < detections.size(); ++i)
            {
                const cv::Point2f detectionMidpoint = center(detections[i]);

                for(unsigned int j = 0; j < imageAndGt.faces.size(); ++j)
                {
                    if ( imageAndGt.faces[j].width * 0.9 <=  detections[i].width && detections[i].width <= imageAndGt.faces[j].width * 1.1
                      && imageAndGt.faces[j].height * 0.9 <=  detections[i].height && detections[i].height <= imageAndGt.faces[j].height * 1.1 )
                    {
                        const cv::Point2f gtMidpoint = center(imageAndGt.faces[j]);

                        if ( imageAndGt.faces[j].width * 0.9 <= std::abs(detectionMidpoint.x - gtMidpoint.x) && std::abs(detectionMidpoint.x - gtMidpoint.x) <= imageAndGt.faces[j].width * 1.1
                          && imageAndGt.faces[j].height * 0.9 <= std::abs(detectionMidpoint.y - gtMidpoint.y) && std::abs(detectionMidpoint.y - gtMidpoint.y) <= imageAndGt.faces[j].height * 1.1 )
                        {
                            ++imageTruePositives;
                            break;
                        }
                    }
                }
            }

            {
                tbb::queuing_mutex::scoped_lock lock(*mutex);
                (*totalTruePositives)  += imageTruePositives;
                (*totalFalsePositives) += (detections.size() - imageTruePositives);
                (*totalScannedWindows) += scannedWindows;

                ++(*evaluatedImages);
                std::cout << "\rProgress " << 100 * (*evaluatedImages) / images->size() << '%';
                std::cout.flush();
                lock.release();
            }
        }
    }
};



template<typename WeakHypothesisType>
int ___main(const std::string testImagesIndexFileName,
            const std::string groundTruthFileName,
            const std::string strongHypothesisFile)
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
    if ( !getTestImages(testImagesIndexFileName, groundTruthFileName, images, totalFacesInGroundTruth) )
    {
        return false;
    }
    std::cout << "Loaded " << images.size() << " images and " << totalFacesInGroundTruth << " ground truth entries." << std::endl;



    int totalTruePositives = 0;
    int totalFalsePositives = 0;

    int totalScannedWindows = 0;
    int evaluatedImages = 0;

    tbb::concurrent_vector<ScannerEntry> entries;

    tbb::queuing_mutex mutex;

    std::cout << "\rProgress " << 100 * (evaluatedImages / images.size()) << '%';
    std::cout.flush();

    tbb::parallel_for(tbb::blocked_range< unsigned int >(0, images.size()),
                      ParallelScan<WeakHypothesisType>(&images,
                                                       &totalScannedWindows,
                                                       &evaluatedImages,
                                                       &strongHypothesis,
                                                       &entries,
                                                       &mutex) );

    std::cout << "\nSorting...\n";
    std::cout.flush();

    tbb::parallel_sort(entries.begin(), entries.end());

    /*
    //USEFULL DEBUG THING!!!
    TotalTruePositivesAndNegatives<WeakHypothesisType> ttpan(
        &images,
        &totalTruePositives,
        &totalFalsePositives,
        &totalScannedWindows,
        &evaluatedImages,
        &mutex,
        &strongHypothesis);
    tbb::blocked_range< unsigned int > range(0, images.size());
    ttpan( range );
    */

    /*
    tbb::parallel_for(tbb::blocked_range< unsigned int >(0, images.size()),
                      TotalTruePositivesAndNegatives<WeakHypothesisType>(
                          &images,
                          &totalTruePositives,
                          &totalFalsePositives,
                          &totalScannedWindows,
                          &evaluatedImages,
                          &mutex,
                          &strongHypothesis) );
    */

    std::cout << "\rTotal subwindows scanned   : " << totalScannedWindows;
    std::cout << "\nTotal faces in ground truth: " << totalFacesInGroundTruth;
    std::cout << "\nTotal true positives       : " << totalTruePositives;
    std::cout << "\nTotal false positives      : " << totalFalsePositives << std::endl;

    return 0;
}

#endif // TEMPLATE_TESTCLASSIFIER_H
