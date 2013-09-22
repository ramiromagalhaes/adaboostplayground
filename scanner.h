#ifndef SCANNER_H
#define SCANNER_H

#include <opencv2/core/core.hpp>
#include "stronghypothesis.h"
#include "labeledexample.h"

template<typename WeakClassifierType>
class Scanner
{
public:
    Scanner(StrongHypothesis<WeakClassifierType> * const classifier_) : classifier(classifier_),
                                                                        scannedWindows(0) {}

    void scan(cv::Mat & image, std::vector<cv::Rect> & detections)
    {
        const int initial_size = 20;
        const float scaling_factor = 1.25f;

        cv::Rect roi(0, 0, initial_size, initial_size);

        for(float scale = 1; scale * initial_size < image.cols && scale * initial_size < image.rows; scale *= scaling_factor)
        {
            const float delta = 1 * scaling_factor;
            roi.width = roi.height = initial_size * scaling_factor;

            for (int x = 0; x < image.cols - roi.width; x += delta)
            {
                roi.x = x;
                for (int y = 0; y < image.rows - roi.height; y += delta)
                {
                    roi.y = y;

                    const cv::Size classifier_size(20, 20);
                    cv::Mat sample(classifier_size, cv::DataType<unsigned char>::type);
                    cv::resize(image(roi), sample, classifier_size/*, 0, 0, INTER_LINEAR*/);

                    const LabeledExample example(sample, no); //classification here is irrelevant.
                                                              //I'm just using it since I didn't yet change
                                                              //things to use Example instead of LabeledExample.

                    if ( classifier->classify(example) == yes )
                    {
                        //TODO integrate detections
                        detections.push_back(roi);
                    }

                    ++scannedWindows;
                }
            }
        }
    }

    unsigned int getScannedWindows()
    {
        return scannedWindows;
    }

private:
    StrongHypothesis<WeakClassifierType> const * const classifier;
    unsigned int scannedWindows;
};

#endif // SCANNER_H
