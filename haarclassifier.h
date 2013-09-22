#ifndef HAARCLASSIFIER_H
#define HAARCLASSIFIER_H

#include <string>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>

#include "common.h"
#include "haarwavelet.h"
#include "labeledexample.h"



template<typename HaarClassifierType>
bool loadHaarClassifiers(const std::string &filename, std::vector<HaarClassifierType> &classifiers)
{
    std::ifstream ifs;
    ifs.open(filename.c_str(), std::ifstream::in);

    if ( !ifs.is_open() )
    {
        return false;
    }

    do
    {
        HaarClassifierType classifier;
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



class HaarClassifier
{
public:
    //About those constructors and operator=, see the links below:
    //http://pages.cs.wisc.edu/~hasti/cs368/CppTutorial/NOTES/CLASSES-PTRS.html#destructor
    //http://stackoverflow.com/questions/6435404/c-error-double-free-or-corruption-fasttop

    HaarClassifier();

    HaarClassifier(HaarWavelet w);

    HaarClassifier(const HaarClassifier & c);

    HaarClassifier & operator=(const HaarClassifier & c);

    virtual ~HaarClassifier();

    virtual bool read(std::istream & in);

    virtual bool write(std::ostream & out) const;

    virtual float featureValue(const Example & example, const float scale = 1.0f) const;

    virtual Classification classify(const Example & example, const float scale = 1.0f) const;

    void setThreshold(const float q_);

    void setPolarity(const float p_);

protected:
    HaarWavelet wavelet;
    float theta;
    float p;
};

class MyHaarClassifier : public HaarClassifier
{
public:
    MyHaarClassifier();

    MyHaarClassifier(HaarWavelet w, std::vector<float> means_);

    MyHaarClassifier(const MyHaarClassifier & c);

    MyHaarClassifier & operator=(const MyHaarClassifier & c);

    virtual ~MyHaarClassifier();

    virtual bool read(std::istream & in);

    virtual bool write(std::ostream & out) const;

    virtual float featureValue(const Example & example, const float scale = 1.0f) const;

    virtual Classification classify(const Example & example, const float scale = 1.0f) const;

protected:
    std::vector<float> means;
    //Here the theta (threshold) will be used as the distance.
    //p is still the polarity
};



#endif // HAARCLASSIFIER_H
