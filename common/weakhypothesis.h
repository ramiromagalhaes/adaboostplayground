#ifndef WEAKHYPOTHESIS_h
#define WEAKHYPOTHESIS_h

#include <string>
#include <sstream>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>

#include <boost/math/distributions/normal.hpp>

#include <haarwavelet.h>
#include <haarwaveletevaluators.h>

#include "common.h"
#include "labeledexample.h"

#define HISTOGRAM_BUCKETS 12

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
class BayesWeakClassifier
{
public:
    BayesWeakClassifier() {}

    BayesWeakClassifier(DualWeightHaarWavelet & f) : feature(f),
                                                     positiveProbability(NormalFeatureValueProbability()),
                                                     negativeProbability(HistogramFeatureValueProbability()) {}

    BayesWeakClassifier(const BayesWeakClassifier & c) : feature(c.feature),
                                                         positiveProbability(c.positiveProbability),
                                                         negativeProbability(c.negativeProbability) {}

    BayesWeakClassifier & operator=(const BayesWeakClassifier & c)
    {
        feature = c.feature;
        positiveProbability = c.positiveProbability;
        negativeProbability = c.negativeProbability;

        return *this;
    }

    ~BayesWeakClassifier() {}

    virtual bool read(std::istream & in)
    {
        return feature.read(in)
                && positiveProbability.read(in)
                && negativeProbability.read(in);
    }

    virtual bool write(std::ostream & out) const
    {
        return feature.write(out)
                && positiveProbability.write(out)
                && negativeProbability.write(out);
    }

    //This is supposed to be used only during trainning and ROC curve construction
    float featureValue(const Example &example, const float scale = 1.0f) const
    {
        const std::pair<float,float> featureValue = evaluator(feature, example.getIntegralSum(), example.getIntegralSquare(), scale);
        return positiveProbability(featureValue.first) - negativeProbability(featureValue.second);
    }

    Classification classify(const Example &example, const float scale = 1.0f) const
    {
        const std::pair<float,float> featureValue = evaluator(feature, example.getIntegralSum(), example.getIntegralSquare(), scale);
        const feature_value_type pos = positiveProbability(featureValue.first);
        const feature_value_type neg = negativeProbability(featureValue.second);
        return pos < neg ? yes : no;
    }

private:

    /**
     * Estimates the probability of a certain feature value being picked using a normal
     * (Gaussian) distribution with the specified mean and standard deviation.
     */
    class NormalFeatureValueProbability {
    public:
        NormalFeatureValueProbability() : distribution(boost::math::normal_distribution<feature_value_type>()),
                                          mean(.0),
                                          stdDev(1.0){}

        NormalFeatureValueProbability(feature_value_type mean_,
                                      feature_value_type stdDev_) : distribution(boost::math::normal_distribution<feature_value_type>()),
                                                                    mean(mean_),
                                                                    stdDev(stdDev_) {}

        bool read(std::istream & in)
        {
            in >> mean
               >> stdDev;

            return true;
        }

        bool write(std::ostream & out) const
        {
            out << ' '
                << mean << ' '
                << stdDev;

            return true;
        }

        feature_value_type operator() (const feature_value_type featureValue) const
        {
            //normalize the featurValue prior to discovering its probability
            return boost::math::pdf(distribution, (featureValue - mean)/stdDev);
        }

    private:
        boost::math::normal_distribution<feature_value_type> distribution;
        feature_value_type mean, stdDev;
    };

    /**
     * Estimates the probability of a certain feature value using a HISTOGRAM_BUCKETS buckets histogram. It is
     * assumed that the feature values ranges from -sqrt(2) to +sqrt(2).
     */
    class HistogramFeatureValueProbability {
    public:
        HistogramFeatureValueProbability() {}
        HistogramFeatureValueProbability(std::vector<feature_value_type> &histogram_)
        {
            histogram = histogram_;
        }

        bool read(std::istream & in)
        {
            int i = 0;
            for (; i < HISTOGRAM_BUCKETS; ++i)
            {
                feature_value_type p;
                in >> p;
                histogram.push_back(p);
            }

            return true;
        }

        bool write(std::ostream & out) const
        {
            int i = 0;
            for (; i < HISTOGRAM_BUCKETS; ++i)
            {
                out << ' ' << histogram[i];
            }

            return true;
        }

        feature_value_type operator() (const feature_value_type featureValue) const
        {
            const int index = featureValue >= std::sqrt(2) ? HISTOGRAM_BUCKETS :
                              featureValue <= -std::sqrt(2) ? 0 :
                              (int)(HISTOGRAM_BUCKETS/2.0 * featureValue / std::sqrt(2)) + HISTOGRAM_BUCKETS/2;
            return histogram[index];
        }

        std::vector<feature_value_type> histogram;
    };



    DualWeightHaarWavelet feature;
    IntensityNormalizedWaveletEvaluator evaluator; //TODO use a static variable
    NormalFeatureValueProbability positiveProbability;
    HistogramFeatureValueProbability negativeProbability;
};


typedef BayesWeakClassifier BayesianHaarClassifier;

#endif // WEAKHYPOTHESIS_h
