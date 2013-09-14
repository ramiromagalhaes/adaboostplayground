#include "haarwavelet.h"
#include <cassert>
#include <limits>



HaarWavelet::HaarWavelet() : scale(1),
                             detectorSize(0),
                             detector(0)
{
}

//TODO verify if all rects_ are under the detector size boundaries...
HaarWavelet::HaarWavelet(cv::Size * const detectorSize_,
                         std::vector<cv::Rect> rects_,
                         std::vector<float> weights_) : scale(1),
                                                        detectorSize(detectorSize_),
                                                        detector(0)
{
    assert(rects.size() == weights.size()); //TODO convert into exception

    rects = rects_;
    weights = weights_;
}

/*
 * Sample wavelet data
 * rects x1 y1 w1 h1 w1 x2 y2 w2 h2 w2...
 *
 */
HaarWavelet::HaarWavelet(cv::Size * const detectorSize_,
                         std::istream & input) : scale(1),
                                                 detectorSize(detectorSize_),
                                                 detector(0)
{
    int rectangles;
    input >> rectangles;

    for (int i = 0; i < rectangles; i++)
    {
        float weight_;
        cv::Rect rect_;

        input >> rect_.x
              >> rect_.y
              >> rect_.width
              >> rect_.height
              >> weight_;

        rects.push_back(rect_);
        weights.push_back(weight_);
    }

}

unsigned int HaarWavelet::dimensions() const
{
    return (int)rects.size();
}

bool HaarWavelet::setIntegralImages(cv::Mat * const sum_, cv::Mat * const squareSum_)
{
    if ( !sum_
            || !squareSum_
            || sum_->cols < detectorSize->width
            || sum_->rows < detectorSize->height)
    {
        return false;
    }

    sum = sum_;
    squareSum = squareSum_;

    return true;
}

//TODO Should the normalization happen here?
float HaarWavelet::value() const
{
    assert(sum && squareSum); //TODO convert into exception?

    float returnValue = 0;

    const int dim = dimensions();
    for (int i = 0; i < dim; ++i)
    {
        float rectValue = singleRectangleValue(rects[i], *sum);
        returnValue += (weights[i] * rectValue);
    }

    return returnValue;
}

void HaarWavelet::srfs(std::vector<float> &srfsVector) const
{
    assert(sum && squareSum); //TODO convert into exception?

    const int dim = dimensions();
    for (int i = 0; i < dim; ++i)
    {
        srfsVector[i] = singleRectangleValue(rects[i], *sum);

        //SRFS works with normalized means (Pavani et al., 2010, section 2.3).
        //AFAIK, Pavani's classifier only normalized things by the maximum numeric value of each pixel.
        //Viola and Jones perform a variance normalization.
        srfsVector[i] /= (rects[i].size().height * rects[i].size().width * std::numeric_limits<unsigned char>::max());
    }
}


/**
 * See also constructor that takes a std::istream.
 */
bool HaarWavelet::write(std::ostream &output) const
{
    if (dimensions() == 0) //won't store a meaningless wavelet
    {
        return false;
    }

    output << dimensions() << ' ';

    bool first = true;
    for (int i = 0; i < dimensions(); i++)
    {
        if (first)
        {
            first = false;
        }
        else
        {
            output << ' ';
        }
        output << rects[i].x << ' '
               << rects[i].y << ' '
               << rects[i].width << ' '
               << rects[i].height << ' '
               << weights[i];
    }

    return true;
}

std::vector<cv::Rect>::const_iterator HaarWavelet::rects_begin() const
{
    return rects.begin();
}

const std::vector<cv::Rect>::const_iterator HaarWavelet::rects_end() const
{
    return rects.end();
}

const cv::Rect HaarWavelet::rect(const int index) const
{
    return rects[index];
}

std::vector<float>::const_iterator HaarWavelet::weights_begin() const
{
    return weights.begin();
}

const std::vector<float>::const_iterator HaarWavelet::weights_end() const
{
    return weights.end();
}

float HaarWavelet::weight(const int index) const
{
    return weights[index];
}

void HaarWavelet::weight(const int index, const float new_value)
{
    weights[index] = new_value;
}

float HaarWavelet::singleRectangleValue(const cv::Rect &r, const cv::Mat &s) const
{
    float rectVal = 0;

    //As per Lienhart, Maydt, 2002, section 2.2
    const int x_w = r.x + r.width;
    const int y_h = r.y + r.height;

    //TODO is there a faster implementation that avoids invoking s.at() functions? Maybe a pointer to the data? Check Mat::step1()
    rectVal = s.at<double>(r.y, r.x)     // (x,     y)
                - s.at<double>(r.y, x_w) // (x + w, y)
                - s.at<double>(y_h, r.x) // (x,     y + h)
                + s.at<double>(y_h, x_w);// (x + w, y + h)

    return rectVal;
}
