#ifndef HAARWAVELET_H
#define HAARWAVELET_H

#include <vector>
#include <iostream>


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Detector;

/**
 * @brief The HaarWavelet class represents a single Haar wavelet.
  */
class HaarWavelet
{
public:

    /**
     * "Raw" constructor for a Haar wavelet.
     */
    HaarWavelet(cv::Size * const detectorSize_, std::vector<cv::Rect> rects_, std::vector<float> weights_);

    /**
     * Constructs a Haar wavelet by reading a input stream.
     */
    HaarWavelet(cv::Size * const detectorSize_, std::istream & input);

    /**
     * Amount of rectangles this Haar wavelet has
     */
    unsigned int dimensions() const;

    //void setDetector(Detector * d_);
    //bool setPosition(cv::Point * pt);

    /**
     * Sets the integral images this Haar Wavelet will use to calculate values. It must receive the
     * reference to the values of the rectangular section over which the Detector is currently
     * iterating over.
     */
    bool setIntegralImages(cv::Mat * const sum_, cv::Mat * const squareSum_/*, cv::Mat * tilted*/); //TODO tilted wavelet

    /**
     * Returns the value of this Haar wavelet when applied to an image in a certain position
     */
    float value() const;

    /**
     * Sets the values of the single rectangle feature space.
     */
    void srfs(std::vector<float> &srfsVector) const;

    /**
     * Writes this Haar wavelet into the given std::ostream.
     */
    bool write(std::ostream &output) const;

    /**
     * Returns a constant iterator to the start of a collection of rectangles that compose this wavelet.
     */
    std::vector<cv::Rect>::const_iterator rects_begin() const;

    /**
     * Returns a constant iterator to the end of the collection of rectangles that compose this wavelet.
     */
    const std::vector<cv::Rect>::const_iterator rects_end() const;

    /**
     * Returns the cv::Rect at position 'index'.
     */
    const cv::Rect rect(const int index) const;

    /**
     * Returns a constant iterator...
     */
    std::vector<float>::const_iterator weights_begin() const;

    /**
     * Returns a constant iterator...
     */
    const std::vector<float>::const_iterator weights_end() const;

    /**
     * Returns the weight applied to a rectangle at position 'index'.
     */
    float weight(const int index) const;

    /**
     * Changes the value of the weight applied to a rectangle at position 'index' of the list of rectangles.
     */
    void weight(const int index, const float new_value);

protected:
    /**
     * Constructs an "empty" instance of this object.
     */
    HaarWavelet();

private:

    /**
     * Calculates the sum of pixels inside a rectangular area of the image.
     */
    float singleRectangleValue(const cv::Rect &r, const cv::Mat &s) const;

    /**
     * Each rectangle and its associated weight of this Haar wavelet
     */
    std::vector<cv::Rect> rects;
    std::vector<float> weights;

    /**
     *If scale > 1, the Haar wavelet streaches right and down
     */
    float scale;

    /**
     * Integral images: both simple sum and square sum. The Haar wavelet only receives the reference
     * to the values of the rectangular section over which the Detector is currently iterating over.
     */
    cv::Mat * sum,
            * squareSum;
    //TODO rotated sum and square sum

    /**
     *Size of the detector window
     */
    cv::Size * const detectorSize; //TODO is this really necessary? Currently it is only used in sanity checks.
                                   //The size and scale can be gathered from the Detector.

    /**
     *Holds a reference for the detector that owns this Haar wavelet.
     */
    Detector * const detector;
};

#endif // HAARWAVELET_H
