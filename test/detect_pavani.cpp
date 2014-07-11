#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "stronghypothesis.h"
#include "weakhypothesis.h"



/**
 * Iterates over an image producing instances of ScannerEntry. Latter, such entries will be processed
 * so a ROC curve is produced.
 */
template<typename WeakClassifierType>
class Scanner
{
public:
    Scanner(StrongHypothesis<WeakClassifierType> & classifier_) : classifier(classifier_),
                                                                  initial_size(20),
                                                                  scaling_factor(1.25),
                                                                  delta(1.5) {}



    void scan(const cv::Mat               & image,
              std::vector<cv::Rect>       & detections)
    {
        cv::Mat integralSum   (image.rows + 1, image.cols + 1, cv::DataType<double>::type);
        cv::Mat integralSquare(image.rows + 1, image.cols + 1, cv::DataType<double>::type);
        cv::integral(image, integralSum, integralSquare, cv::DataType<double>::type);

        //This algorithm will iterate over the INTEGRAL images, reflecting what would be happening while
        //iterating over the real image.

        for(double scale = 1.5; scale * initial_size < integralSum.cols
                             && scale * initial_size < integralSum.rows; scale *= scaling_factor)
        {
            const double shift = delta * scale;
            cv::Rect integralRoi(0, 0, (initial_size * scale) + 1, (initial_size * scale) + 1); //The integral image ROI is 1 unit bigger than the original image ROI.

            for (integralRoi.x = 0; integralRoi.x <= integralSum.cols - integralRoi.width; integralRoi.x += shift)
            {
                for (integralRoi.y = 0; integralRoi.y <= integralSum.rows - integralRoi.height; integralRoi.y += shift)
                {
                    cv::Rect roi = integralRoi; //This is the ROI on the real image. We need it because the
                    roi.width -= 1;             //integral images ROIs are 1 unit bigger the the original.
                    roi.height -= 1;            //This unit shouldn't be used when scaling the real image ROI.

                    const Example example(integralSum(integralRoi), integralSquare(integralRoi));

                    if ( classifier.classify(example, scale) == yes )
                    {
                        detections.push_back(roi);
                    }
                }
            }
        }
    }

private:

    const StrongHypothesis<WeakClassifierType> & classifier;
    const int initial_size;      //initial width and height of the detector
    const double scaling_factor; //how mutch the scale will change per iteration
    const double delta;          //window shift constant
};



/**
 *
 */
int main(int, char **argv) {
    const std::string imagePath = argv[1];
    const std::string classifierPath = argv[2];
    const std::string thresholdParam = argv[3];



    cv::Mat image = cv::imread(imagePath, cv::DataType<unsigned char>::type);
    if ( !image.data )
    {
        return 2;
    }



    StrongHypothesis<PavaniHaarClassifier> strongHypothesis;
    {
        std::ifstream in(classifierPath.c_str());
        if ( !in.is_open() )
        {
            return 7;
        }
        if ( !strongHypothesis.read(in) )
        {
            return 11;
        }

        double threshold;
        std::stringstream stringstream(thresholdParam);
        stringstream >> threshold;
        strongHypothesis.setThreshold(threshold);

        std::cout << "Loaded strong classifier from " << classifierPath << " with threshold " << threshold << '.' << std::endl;
    }



    std::vector<cv::Rect> detections;
    Scanner<PavaniHaarClassifier> scanner(strongHypothesis);
    scanner.scan(image, detections);
    std::cout << "Found " << detections.size() << " face(s)." << std::endl;



    for(std::vector<cv::Rect>::iterator it = detections.begin(); it != detections.end(); ++it)
    {
        cv::rectangle(image, *it, cv::Scalar( 255, 0, 0 ));
    }
    cv::imshow("Detections", image);
    cv::waitKey( 0 );



    return 0;
}
