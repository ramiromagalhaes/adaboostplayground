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



struct ImageAndGroundTruth
{
    cv::Mat image;
    std::vector<cv::Rect> faces;
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



cv::Point2f center(cv::Rect & rect)
{
    return cv::Point(rect.x + rect.width/2.0f, rect.y + rect.height/2.0f);
}



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
    }


    int totalFacesInGroundTruth = 0;
    std::vector<ImageAndGroundTruth> images;
    if ( !getTestImages(testImagesIndexFileName, groundTruthFileName, images, totalFacesInGroundTruth) )
    {
        return false;
    }

    int truePositives = 0;
    int falsePositives = 0;

    Scanner<WeakHypothesisType> scanner(&strongHypothesis);

    for(std::vector<ImageAndGroundTruth>::iterator imageAndGt = images.begin(); imageAndGt != images.end(); ++imageAndGt)
    {
        std::cout << "\rProgress " << 100 * (imageAndGt - images.begin()) / images.size() << '%';
        std::cout.flush();

        std::vector<cv::Rect> detections(0);
        scanner.scan(imageAndGt->image, detections);

        for(unsigned int i = 0; i < detections.size(); ++i)
        {
            const cv::Point2f detectionMidpoint = center(detections[i]);

            for(unsigned int j = 0; j < imageAndGt->faces.size(); ++j)
            {
                if ( imageAndGt->faces[j].width * 0.9 <=  detections[i].width && detections[i].width <= imageAndGt->faces[j].width * 1.1
                  && imageAndGt->faces[j].height * 0.9 <=  detections[i].height && detections[i].height <= imageAndGt->faces[j].height * 1.1 )
                {
                    const cv::Point2f gtMidpoint = center(imageAndGt->faces[j]);

                    if ( imageAndGt->faces[j].width * 0.9 <= std::abs(detectionMidpoint.x - gtMidpoint.x) && std::abs(detectionMidpoint.x - gtMidpoint.x) <= imageAndGt->faces[j].width * 1.1
                      && imageAndGt->faces[j].height * 0.9 <= std::abs(detectionMidpoint.y - gtMidpoint.y) && std::abs(detectionMidpoint.y - gtMidpoint.y) <= imageAndGt->faces[j].height * 1.1 )
                    {
                        ++truePositives;
                        break;
                    }
                }
            }
        }

        falsePositives += (detections.size() - truePositives);
    }

    std::cout <<   "Total subwindows scanned   : " << scanner.getScannedWindows();
    std::cout << "\nTotal faces in ground truth: " << totalFacesInGroundTruth;
    std::cout << "\nTotal true positives       : " << truePositives;
    std::cout << "\nTotal false positives      : " << falsePositives << std::endl;

    /*
    std::vector<RocRecord> records(samples.size());
    tbb::parallel_for( tbb::blocked_range< unsigned int >(0, samples.size()),
                       RocCalculator<WeakHypothesisType>(&samples, &records, &strongHypothesis) );

    std::sort(records.begin(), records.end());

    for (unsigned int i = 0; i < samples.size(); ++i)
    {
        std::cout << records[i].falsePositives << ' ' << records[i].truePositives << std::endl;
    }
    */

    return 0;
}

#endif // TEMPLATE_TESTCLASSIFIER_H
