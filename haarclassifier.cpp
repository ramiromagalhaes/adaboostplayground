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



HaarClassifier::~HaarClassifier()
{
    /* TODO properly delete the reference to wavelet. This is currently not working. No idea why.
    if (wavelet != 0)
    {
        delete wavelet;
        wavelet = 0;
    }
    */
}



bool HaarClassifier::read(std::istream & in)
{
    //whatever's not about the HaarWavelet will be discarded
    wavelet.read(in);

    std::vector<float> mean(wavelet.dimensions());

    for (unsigned int i = 0; i < wavelet.dimensions(); i++)
    {
        in >> mean[i];
    }

    float stdDev, q;
    in >> stdDev;
    in >> q;

    return true;
}



bool HaarClassifier::write(std::ostream & out) const
{
    //whatever's not about the HaarWavelet won't be written
    if ( !wavelet.write(out) )
    {
        return false;
    }

    for (unsigned int i = 0; i < wavelet.dimensions(); i++)
    {
        out << ' ' << 0;
    }

    out << ' '
        << p << ' '
        << theta;

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
    return do_classify( wavelet.value(example.getIntegralSum(), example.getIntegralSquare()) );
}

Classification HaarClassifier::do_classify(const float f) const
{

    return f * p <= theta * p ? yes : no;
}



bool HaarClassifier::loadClassifiers(const std::string &filename, std::vector<HaarClassifier> & classifiers)
{
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ifstream::in);

    if ( !ifs.is_open() )
    {
        return false;
    }

    do
    {
        HaarClassifier classifier;
        classifier.read(ifs);

        if ( !ifs.eof() )
        {
            classifiers.push_back( classifier );
        }
        else
        {
            break;
        }
    } while (true);

    ifs.close();

    return true;
}
