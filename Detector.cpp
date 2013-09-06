#include "Detector.h"



Detector::Detector()
{
}



bool Detector::setSubwindow(const int left, const int top)
{
    if( left < 0 || top < 0 ||
        left + defaultSize.width >= image.cols ||
        top + defaultSize.height >= image.rows )
    {
        return false;
    }

    currentRoi.x = left;
    currentRoi.y = top;

    {
        const cv::Rect integralsRoi(currentRoi.x, currentRoi.y, currentRoi.width + 1, currentRoi.height + 1);
        currentIntegralImage = integralImage(integralsRoi);
        currentSquareIntegralImage = squareIntegralImage(integralsRoi);
    }

    const double area = currentRoi.area();
    subwindowMean = (currentIntegralImage.at<int>(0, 0)
                    - currentIntegralImage.at<int>(0, currentRoi.width)
                    - currentIntegralImage.at<int>(currentRoi.height, 0)
                    + currentIntegralImage.at<int>(currentRoi.height, currentRoi.width)) / area;

    subwindowStdDeviation = sqrt(
        subwindowMean * subwindowMean - (currentSquareIntegralImage.at<int>(0, 0)
                                            - currentSquareIntegralImage.at<int>(0, currentRoi.width)
                                            - currentSquareIntegralImage.at<int>(currentRoi.height, 0)
                                            + currentSquareIntegralImage.at<int>(currentRoi.height, currentRoi.width)
                                         ) / area
    );

    //snipet taken from OpenCV. This is how they calculate the "normalization factor"
    //return (float) sqrt( (double) (area * valSqSum - (double)valSum * valSum) );

//    const double normalized = (pixel - subwindowMean)/sqrt(subwindowVariance);
}



bool Detector::setSubwindow(const cv::Point topLeftPosition)
{
    setSubwindow(topLeftPosition.x, topLeftPosition.y);
}