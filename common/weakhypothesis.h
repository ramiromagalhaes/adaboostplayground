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



/**
 * Loads many WeakHypothesis found in a file to a vector of HaarClassifierType.
 */
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



/**
 * Template for a weak classifier that uses a Haar-like feature.
 */
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



    //This is supposed to be used only during trainning and ROC curve construction
    float featureValue(const Example &example, const float scale = 1.0f) const
    {
        return evaluator(feature, example.getIntegralSum(), example.getIntegralSquare(), scale);
    }

    Classification classify(const Example &example, const float scale = 1.0f) const
    {
        return featureValue(example, scale) * p <= theta * p ? yes : no;
    }

private:
    FeatureType feature;
    HaarEvaluatorType evaluator; //TODO use a static variable
    float theta;
    float p;
};



//Viola & Jones' classifier
typedef ThresholdedWeakClassifier<HaarWavelet, VarianceNormalizedWaveletEvaluator> ViolaJonesClassifier;

//Pavani's classifier.
typedef ThresholdedWeakClassifier<HaarWavelet, IntensityNormalizedWaveletEvaluator> PavaniHaarClassifier;

//My classifier
typedef ThresholdedWeakClassifier<MyHaarWavelet, IntensityNormalizedWaveletEvaluator> MyHaarClassifier;



/**
 * A classifier that uses as classification criteria the naive Bayes rule.
 */
template <typename FeatureType, typename PositiveHaarEvaluatorType, typename NegativeHaarEvaluatorType> //TODO do not use TEMPLATES; use a Strategy instead.
class NaiveBayesWeakClassifier
{
public:
    NaiveBayesWeakClassifier() {}

    NaiveBayesWeakClassifier(FeatureType & f, PositiveHaarEvaluatorType posEval, NegativeHaarEvaluatorType negEval) : feature(f),
                                                                                                                      positiveEvaluator(posEval),
                                                                                                                      negativeEvaluator(negEval) {}

    NaiveBayesWeakClassifier(const NaiveBayesWeakClassifier & c) : feature(c.feature),
                                                                   positiveEvaluator(c.positiveEvaluator),
                                                                   negativeEvaluator(c.negativeEvaluator) {}

    NaiveBayesWeakClassifier & operator=(const NaiveBayesWeakClassifier & c)
    {
        feature = c.feature;
        positiveEvaluator = c.positiveEvaluator;
        negativeEvaluator = c.negativeEvaluator;

        return *this;
    }

    ~NaiveBayesWeakClassifier() {}

    virtual bool read(std::istream & in)
    {
        return feature.read(in)
                && positiveEvaluator.read(in)
                && negativeEvaluator(in);
    }

    virtual bool write(std::ostream & out) const
    {
        return feature.write(out)
                && positiveEvaluator.write(out)
                && negativeEvaluator(out);
    }

    //This is supposed to be used only during trainning and ROC curve construction
    float featureValue(const Example &example, const float scale = 1.0f) const
    {
        //TODO review me!!!
        return positiveEvaluator(example, scale) - negativeEvaluator(example, scale);
    }

    Classification classify(const Example &example, const float scale = 1.0f) const
    {
        return positiveEvaluator(example, scale) < negativeEvaluator(example, scale) ? yes : no;
    }

private:
    FeatureType feature;
    PositiveHaarEvaluatorType positiveEvaluator;
    NegativeHaarEvaluatorType negativeEvaluator;


    class PositiveHaarEvaluator {
    public:
        PositiveHaarEvaluator() {}
        PositiveHaarEvaluator(std::vector<feature_value_type> & mean__,
                              cv::Mat< cv::DataType<feature_value_type>::type > & covarMatrix__)
        {
            mean = mean__;
            covarMatrix = covarMatrix__;
        }

        operator() (Example &example, const float scale = 1.0f) const
        {
            //TODO get SRFS value from example and scale
            //TODO get its probability to happen. consider boost: http://www.boost.org/doc/libs/1_53_0/libs/math/doc/sf_and_dist/html/
        }

    private:
        std::vector<feature_value_type> mean;
        cv::Mat< cv::DataType<feature_value_type>::type > covarMatrix; //TODO consider using BOOST statistics module?
    };

    class NegativeHaarEvaluator {
    public:
        NegativeHaarEvaluator() {}
        NegativeHaarEvaluator()
        {
            //Sets the histogram
        }

        operator() (Example &example, const float scale = 1.0f) const
        {
            //TODO get SRFS value from example and scale
            //TODO create a hash from the SRFS value to seek for its probability in the histogram table.
        }
    };

};



#endif // WEAKHYPOTHESIS_h
