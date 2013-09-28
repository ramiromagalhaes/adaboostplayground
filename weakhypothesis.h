#ifndef WEAKHYPOTHESIS_h
#define WEAKHYPOTHESIS_h

#include <string>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>

#include <haarwavelet.h>

#include "common.h"
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


template <typename FeatureType>
class ThresholdedWeakClassifier
{
public:
    //About those constructors and operator=, see the links below:
    //http://pages.cs.wisc.edu/~hasti/cs368/CppTutorial/NOTES/CLASSES-PTRS.html#destructor
    //http://stackoverflow.com/questions/6435404/c-error-double-free-or-corruption-fasttop

    ThresholdedWeakClassifier() : feature(),
                                  theta(0),
                                  p(1) {}

    ThresholdedWeakClassifier(FeatureType & f) : feature(f),
                                                 theta(0),
                                                 p(1) {}

    ThresholdedWeakClassifier(const ThresholdedWeakClassifier & c) : feature(c.feature),
                                                                     theta(c.theta),
                                                                     p(c.p) {}

    ThresholdedWeakClassifier & operator=(const ThresholdedWeakClassifier & c)
    {
        feature = c.feature;
        theta = c.theta;
        p = c.p;

        return *this;
    }

    ~ThresholdedWeakClassifier() {}



    virtual bool read(std::istream & in)
    {
        if ( !feature.read(in) )
        {
            return false;
        }

        in >> p
           >> theta;

        return true;
    }

    virtual bool write(std::ostream & out) const
    {
        if ( !feature.write(out) )
        {
            return false;
        }

        out << ' '
            << p << ' '
            << theta;

        return true;
    }



    void setThreshold(const float theta_)
    {
        theta = theta_;
    }

    void setPolarity(const float p_)
    {
        p = p_;
    }



    //This is supposed to be used only during trainning
    float featureValue(const Example &example, const float scale = 1.0f) const
    {
        return feature.value(example.getIntegralSum(), example.getIntegralSquare(), scale);
    }


    Classification classify(const Example &example, const float scale = 1.0f) const
    {
        return featureValue(example, scale) * p <= theta * p ? yes : no;
    }

protected:
    FeatureType feature;
    float theta;
    float p;
};



typedef ThresholdedWeakClassifier<HaarWavelet> HaarClassifier;
typedef ThresholdedWeakClassifier<MyHaarWavelet> MyHaarClassifier;
typedef ThresholdedWeakClassifier<ViolaJonesHaarWavelet> ViolaJonesClassifier;




#endif // WEAKHYPOTHESIS_h
