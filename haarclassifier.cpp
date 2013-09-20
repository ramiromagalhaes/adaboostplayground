#include "haarclassifier.h"

#include <numeric>
#include <opencv2/core/core.hpp>



HaarClassifier::HaarClassifier() : wavelet(),
                                   theta(0),
                                   p(1) {}

HaarClassifier::HaarClassifier(HaarWavelet w) : wavelet(w),
                                                theta(0),
                                                p(1) {}

HaarClassifier::HaarClassifier(const HaarClassifier & c) : wavelet(c.wavelet),
                                                           theta(c.theta),
                                                           p(c.p) {}


HaarClassifier & HaarClassifier::operator=(const HaarClassifier & c)
{
    wavelet = c.wavelet;
    theta = c.theta;
    p = c.p;

    return *this;
}


HaarClassifier::~HaarClassifier() {}



bool HaarClassifier::read(std::istream & in)
{
    wavelet.read(in);
    in >> p
       >> theta;

    return true;
}

bool HaarClassifier::write(std::ostream & out) const
{
    //whatever's not about the HaarWavelet won't be written
    if ( !wavelet.write(out) )
    {
        return false;
    }

    out << ' '
        << p << ' '
        << theta;

    for (unsigned int i = 0; i < wavelet.dimensions(); i++)
    {
        out << ' ' << 0;
    }

    return true;
}



void HaarClassifier::setThreshold(const float theta_)
{
    theta = theta_;
}

void HaarClassifier::setPolarity(const float p_)
{
    p = p_;
}



//This is supposed to be used only during trainning
float HaarClassifier::featureValue(const LabeledExample &example) const
{
    return wavelet.value(example.getIntegralSum(), example.getIntegralSquare());
}


Classification HaarClassifier::classify(const LabeledExample &example) const
{
    return featureValue(example) * p <= theta * p ? yes : no;
}



//============================== MyHaarClassifier ==============================



MyHaarClassifier::MyHaarClassifier() : HaarClassifier(),
                                       means(0) {}

MyHaarClassifier::MyHaarClassifier(HaarWavelet w, std::vector<float> means_) : HaarClassifier(w),
                                                                                means(means_) {}

MyHaarClassifier::MyHaarClassifier(const MyHaarClassifier &c) : HaarClassifier()
{
    wavelet = c.wavelet;
    theta = c.theta;
    p = c.p;
    means = c.means;
}

MyHaarClassifier &MyHaarClassifier::operator=(const MyHaarClassifier &c)
{
    wavelet = c.wavelet;
    theta = c.theta;
    p = c.p;
    means = c.means;

    return *this;
}

MyHaarClassifier::~MyHaarClassifier() {}

bool MyHaarClassifier::read(std::istream &in)
{
    if ( !HaarClassifier::read(in) )
    {
        return false;
    }

    means.resize(wavelet.dimensions());
    for (unsigned int i = 0; i < wavelet.dimensions(); i++)
    {
        in >> means[i];
    }

    return true;
}

bool MyHaarClassifier::write(std::ostream &out) const
{
    if ( !HaarClassifier::write(out) )
    {
        return false;
    }

    for (unsigned int i = 0; i < wavelet.dimensions(); i++)
    {
        out << ' ' << means[i];
    }

    return true;
}

float MyHaarClassifier::featureValue(const LabeledExample &example) const
{
    std::vector<float> s(wavelet.dimensions());
    wavelet.srfs(example.getIntegralSum(), example.getIntegralSquare(), s);

    float distance = 0;
    for(unsigned int i = 0; i < s.size(); ++i)
    {
        const float v = s[i] - means[i];
        distance += v * v;
    }

    return std::sqrt(distance);
}

Classification MyHaarClassifier::classify(const LabeledExample &example) const
{
    const float f = featureValue(example);

    if (p == 1)
    {
        return  theta <= f && f <= theta ? yes : no;
    }
    else if (p == -1)
    {
        return  theta <= f && f <= theta ? no : yes;
    }
    else
    {
        throw 201; //p is supposed to be +1 or -1
    }
}
