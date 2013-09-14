#include "haarclassifier.h"

#include <limits>
#include <opencv2/core/core.hpp>



HaarClassifier::HaarClassifier() : wavelet(0),
                                   mean(2),
                                   stdDev(1),
                                   q(2) {}



HaarClassifier::HaarClassifier(HaarWavelet *w) : wavelet(w),
                                                 mean(2),
                                                 stdDev(1),
                                                 q(2) {}



HaarClassifier::HaarClassifier(const HaarClassifier & c) : wavelet(c.wavelet),
                                                           mean(c.mean),
                                                           stdDev(c.stdDev),
                                                           q(c.q) {}



HaarClassifier & HaarClassifier::operator=(const HaarClassifier & c)
{
    wavelet = c.wavelet;
    mean = c.mean;
    stdDev = c.stdDev;
    q = c.q;

    return *this;
}



HaarClassifier::~HaarClassifier()
{
    /* TODO properly delete the reference to wavelet. It is currently not working
    if (wavelet != 0)
    {
        delete wavelet;
        wavelet = 0;
    }
    */
}



bool HaarClassifier::read(std::istream & in)
{
    wavelet = new HaarWavelet(new cv::Size(20, 20), in);

    mean.resize(wavelet->dimensions());

    for (unsigned int i = 0; i < wavelet->dimensions(); i++)
    {
        in >> mean[i];
    }

    in >> stdDev;
    float discard_q;
    in >> discard_q;

    return true;
}



bool HaarClassifier::write(std::ostream & out) const
{
    if ( !wavelet->write(out) )
    {
        return false;
    }

    for (unsigned int i = 0; i < mean.size(); i++)
    {
        out << ' ' << mean[i];
    }

    out << ' '
        << stdDev << ' '
        << q;

    return true;
}



Classification HaarClassifier::classify(LabeledExample & example) const
{
    wavelet->setIntegralImages(&example.integralSum, &example.integralSquare);

    std::vector<float> s(wavelet->dimensions());
    wavelet->srfs(s);

    float distance = 0;
    for(unsigned int i = 0; i < s.size(); ++i)
    {
        const float v = s[i] - mean[i];
        distance += v * v;
    }
    distance = std::sqrt(distance);

    if (distance < stdDev)
    {
        return yes;
    }

    return no;
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
