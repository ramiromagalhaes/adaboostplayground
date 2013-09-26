#include "haarclassifier.h"

#include <numeric>
#include <opencv2/core/core.hpp>



HaarClassifier::HaarClassifier() : wavelet(),
                                   theta(0),
                                   p(1) {}

HaarClassifier::HaarClassifier(HaarWavelet & w) : wavelet(w),
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
    if ( !wavelet.read(in) )
    {
        return false;
    }

    in >> p
       >> theta;

    for (unsigned int i = 0; i < wavelet.dimensions(); i++)
    {
        float mean;
        in >> mean; //discard the means.
    }

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
float HaarClassifier::featureValue(const Example &example, const float scale) const
{
    return wavelet.value(example.getIntegralSum(), example.getIntegralSquare(), scale);
}


Classification HaarClassifier::classify(const Example &example, const float scale) const
{
    return featureValue(example, scale) * p <= theta * p ? yes : no;
}



//============================== MyHaarClassifier ==============================



MyHaarClassifier::MyHaarClassifier() : wavelet(),
                                       theta(0),
                                       p(1) {}

MyHaarClassifier::MyHaarClassifier(MyHaarWavelet & w, std::vector<float> means_) : wavelet(w),
                                                                                   theta(0),
                                                                                   p(1) {}

MyHaarClassifier::MyHaarClassifier(const MyHaarClassifier &c)
{
    wavelet = c.wavelet;
    theta = c.theta;
    p = c.p;
}

MyHaarClassifier &MyHaarClassifier::operator=(const MyHaarClassifier &c)
{
    wavelet = c.wavelet;
    theta = c.theta;
    p = c.p;

    return *this;
}

MyHaarClassifier::~MyHaarClassifier() {}



bool MyHaarClassifier::read(std::istream &in)
{
    if ( !wavelet.read(in) )
    {
        return false;
    }

    in >> p
       >> theta;

    return true;
}

bool MyHaarClassifier::write(std::ostream &out) const
{
    if ( !wavelet.write(out) )
    {
        return false;
    }

    out << ' '
        << p << ' '
        << theta;

    return true;
}



float MyHaarClassifier::featureValue(const Example &example, const float scale) const
{
    return wavelet.value(example.getIntegralSum(), example.getIntegralSquare(), scale);
}

Classification MyHaarClassifier::classify(const Example &example, const float scale) const
{
    return  p * featureValue(example, scale) <= theta * p? yes : no;
}



void MyHaarClassifier::setThreshold(const float theta_)
{
    theta = theta_;
}

void MyHaarClassifier::setPolarity(const float p_)
{
    p = p_;
}



//============================== ViolaJonesClassifier ==============================



ViolaJonesClassifier::ViolaJonesClassifier() : wavelet(),
                                   theta(0),
                                   p(1) {}

ViolaJonesClassifier::ViolaJonesClassifier(ViolaJonesHaarWavelet & w) : wavelet(w),
                                                                        theta(0),
                                                                        p(1) {}

ViolaJonesClassifier::ViolaJonesClassifier(const ViolaJonesClassifier & c) : wavelet(c.wavelet),
                                                           theta(c.theta),
                                                           p(c.p) {}


ViolaJonesClassifier & ViolaJonesClassifier::operator=(const ViolaJonesClassifier & c)
{
    wavelet = c.wavelet;
    theta = c.theta;
    p = c.p;

    return *this;
}


ViolaJonesClassifier::~ViolaJonesClassifier() {}



bool ViolaJonesClassifier::read(std::istream & in)
{
    wavelet.read(in);
    in >> p
       >> theta;

    //TODO remove that when you change the write method
    for (unsigned int i = 0; i < wavelet.dimensions(); i++)
    {
        float discard;
        in >> discard;
    }

    return true;
}

bool ViolaJonesClassifier::write(std::ostream & out) const
{
    if ( !wavelet.write(out) )
    {
        return false;
    }

    out << ' '
        << p << ' '
        << theta;

    //TODO No need to write that on the strong classifier should remove.
    for (unsigned int i = 0; i < wavelet.dimensions(); i++)
    {
        out << ' ' << 0;
    }

    return true;
}



void ViolaJonesClassifier::setThreshold(const float theta_)
{
    theta = theta_;
}

void ViolaJonesClassifier::setPolarity(const float p_)
{
    p = p_;
}



//This is supposed to be used only during trainning
float ViolaJonesClassifier::featureValue(const Example &example, const float scale) const
{
    return wavelet.value(example.getIntegralSum(), example.getIntegralSquare(), scale);
}


Classification ViolaJonesClassifier::classify(const Example &example, const float scale) const
{
    return featureValue(example, scale) * p <= theta * p ? yes : no;
}
