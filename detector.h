#ifndef DETECTOR_H
#define DETECTOR_H

#include "haarwavelet.h"

class Detector
{
public:
    Detector();

    bool setSubwindow(const int left, const int top);
    bool setSubwindow(const cv::Point topLeftPosition);

private:
    /**
     * @brief image the whole image over which the detector iterates over.
     */
    const cv::Mat image;

    /**
     * @brief image the integral image calculated over the image.
     */
    const cv::Mat integralImage;

    /**
     * @brief image the square integral image calculated over the image.
     */
    const cv::Mat squareIntegralImage;

    /**
     * @brief defaultSize the size of the detector when it is scaled by 1. Typically it is the size the detector is supposed to be during trainning.
     */
    const cv::Size defaultSize;

    /**
     * @brief currentRoi stores both the current size and position of the scanning window.
     */
    cv::Rect currentRoi;

    /**
     * @brief currentSubwindow references the integral image data that's being scanned.
     */
    cv::Mat currentIntegralImage;

    /**
     * @brief currentSubwindow references the square integral image data that's being scanned.
     */
    cv::Mat currentSquareIntegralImage;

    /**
     * @brief subwindowMean The mean value of the pixels in the subwindow.
     */
    float subwindowMean;

    /**
     * @brief subwindowStdDeviation The standard deviation of the pixels in the subwindow.
     */
    float subwindowStdDeviation;

};

#endif // DETECTOR_H
