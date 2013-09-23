#ifndef SCANNER_H
#define SCANNER_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "stronghypothesis.h"
#include "labeledexample.h"

template<typename WeakClassifierType>
class Scanner
{
public:
    Scanner(StrongHypothesis<WeakClassifierType> * const classifier_) : classifier(classifier_),
                                                                        initial_size(20),
                                                                        scaling_factor(1.25f),
                                                                        delta(1.5f) {}

    unsigned int scan(cv::Mat & image, std::vector<cv::Rect> & detections)
    {
        cv::Mat integralSum   (image.rows + 1, image.cols + 1, cv::DataType<double>::type);
        cv::Mat integralSquare(image.rows + 1, image.cols + 1, cv::DataType<double>::type);
        cv::integral(image, integralSum, integralSquare, cv::DataType<double>::type);

        const float max_scaling_factor = std::pow(scaling_factor, 5.0f);

        unsigned int scannedWindows = 0;

        //We iterate our ROI over the integral image, therefore, when we detect something,
        //we must set the correct ROI.
        cv::Rect roi(0, 0, initial_size, initial_size);

        for(float scale = 1; scale * initial_size < image.cols
                          && scale * initial_size < image.rows
                          && scale < max_scaling_factor; scale *= scaling_factor)
        {
            const float shift = delta * scale;
            roi.width = roi.height = initial_size * scale + 1;

            for (roi.x = 1; roi.x < integralSum.cols - roi.width; roi.x += shift)
            {
                for (roi.y = 1; roi.y < integralSum.rows - roi.height; roi.y += shift)
                {
                    cv::Rect exampleRoi = roi;
                    --exampleRoi.x;
                    --exampleRoi.y;
                    const Example example(integralSum(exampleRoi), integralSquare(exampleRoi));

                    if ( classifier->classify(example, scale) == yes )
                    {
                        detections.push_back(roi);
                    }

                    ++scannedWindows;
                }
            }
        }

        integrateDetection(detections);

        return scannedWindows;
    }



private:
    StrongHypothesis<WeakClassifierType> const * const classifier;
    const int initial_size; //initial width and height of the detector
    const float scaling_factor; //how mutch the scale will change per iteration
    const float delta; //window shift constant

    //As described by Viola & Jones 2004.
    //This is an awful slow implementation. Find a way to speed it up!!!
    void integrateDetection(std::vector<cv::Rect> & detections)
    {
        std::vector< std::vector<cv::Rect> > subsets;

        for(unsigned int i = 0; i < detections.size(); ++i)
        {
            bool inserted = false;

            for(unsigned int j = 0; j < subsets.size(); ++j)
            {
                for(unsigned int k = 0; k < subsets[j].size(); ++k)
                {
                    if ( (detections[i] & subsets[j][k]) != cv::Rect(0,0,0,0) )
                    {
                        subsets[j].push_back(detections[i]);
                        inserted = true;
                        break;
                    }
                }

                if (inserted)
                {
                    break;
                }
            }

            if (!inserted)
            {
                subsets.push_back(std::vector<cv::Rect>());
                subsets[subsets.size() - 1].push_back(detections[i]);
            }
        }

        detections.clear();

        for(unsigned int i = 0; i < subsets.size(); ++i)
        {
            double integratedX = 0;
            double integratedY = 0;
            double integratedWidth = 0;
            double integratedHeight = 0;

            for(unsigned int j = 0; j < subsets[i].size(); ++j)
            {
                integratedX += subsets[i][j].x;
                integratedY += subsets[i][j].y;
                integratedWidth  += subsets[i][j].width;
                integratedHeight += subsets[i][j].height;
            }

            integratedX      /= subsets[i].size();
            integratedY      /= subsets[i].size();
            integratedWidth  /= subsets[i].size();
            integratedHeight /= subsets[i].size();

            detections.push_back(
                cv::Rect(integratedX,
                         integratedY,
                         integratedWidth,
                         integratedHeight));
        }
    }

};

#endif // SCANNER_H
