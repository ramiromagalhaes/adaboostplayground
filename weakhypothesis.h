#ifndef WEAKHYPOTHESIS_h
#define WEAKHYPOTHESIS_h

#include <string>
#include <sstream>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>

#include <haarwavelet.h>
#include <haarwaveletevaluators.h>

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
        std::string line;
        getline(ifs, line);
        if (line.empty())
        {
            break;
        }
        std::istringstream lineInputStream(line);

        HaarClassifierType classifier;
        classifier.read(lineInputStream);

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



template <typename FeatureType, typename HaarEvaluatorType>
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
        return evaluator(feature, example.getIntegralSum(), example.getIntegralSquare(), scale);
    }

    Classification classify(const Example &example, const float scale = 1.0f) const
    {
        return featureValue(example, scale) * p <= theta * p ? yes : no;
    }

protected:
    FeatureType feature;
    HaarEvaluatorType evaluator; //TODO use a static variable
    float theta;
    float p;
};



//Viola & Jones' classifier
typedef ThresholdedWeakClassifier<HaarWavelet, VarianceNormalizedWaveletEvaluator> ViolaJonesClassifier;

//Pavani's classifier.
typedef ThresholdedWeakClassifier<HaarWavelet, IntensityNormalizedWaveletEvaluator> PavaniHaarClassifier;
typedef ThresholdedWeakClassifier<HaarWavelet, VarianceNormalizedWaveletEvaluator>  PavaniVarNormHaarClassifier;

//My classifier
typedef ThresholdedWeakClassifier<MyHaarWavelet, IntensityNormalizedWaveletEvaluator> MyHaarClassifier;
typedef ThresholdedWeakClassifier<MyHaarWavelet, VarianceNormalizedWaveletEvaluator>  MyVarNormHaarClassifier;



#endif // WEAKHYPOTHESIS_h
