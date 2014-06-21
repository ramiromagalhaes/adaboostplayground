#ifndef WEAKHYPOTHESIS_h
#define WEAKHYPOTHESIS_h

#include <string>
#include <sstream>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/laplace.hpp>

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
 * Estimates the probability of a certain feature value being picked using a normal
 * (Gaussian) distribution with the specified mean and standard deviation.
 */
class NormalFeatureValueProbability {
public:
    NormalFeatureValueProbability() : prior(1.0),
                                      mean(.0),
                                      stdDev(1.0){}

    NormalFeatureValueProbability(feature_value_type prior_,
                                  feature_value_type mean_,
                                  feature_value_type stdDev_) : distribution(boost::math::normal_distribution<feature_value_type>()),
                                                                prior(prior_),
                                                                mean(mean_),
                                                                stdDev(stdDev_) {}

    bool read(std::istream & in)
    {
        in >> prior
           >> mean
           >> stdDev;
        return true;
    }

    bool write(std::ostream & out) const
    {
        out << ' '
            << prior << ' '
            << mean << ' '
            << stdDev;
        return true;
    }

    feature_value_type operator() (const feature_value_type featureValue) const
    {
        return prior
                * boost::math::pdf(distribution,
                                   (featureValue - mean)/stdDev); //normalize the featureValue before calculating its probability
    }

private:
    boost::math::normal_distribution<feature_value_type> distribution;
    feature_value_type prior, mean, stdDev;
};



/**
 *
 */
class LaplaceFeatureValueProbability {
public:
    LaplaceFeatureValueProbability() : prior(1.0),
                                       mean(.0),
                                       stdDev(1.0) {}

    LaplaceFeatureValueProbability(feature_value_type prior_,
                                   feature_value_type mean_,
                                   feature_value_type stdDev_) : prior(prior_),
                                                                mean(mean_),
                                                                stdDev(stdDev_) {}

    bool read(std::istream & in)
    {
        in >> prior
           >> mean
           >> stdDev;
        return true;
    }

    bool write(std::ostream & out) const
    {
        out << ' '
            << prior << ' '
            << mean << ' '
            << stdDev;
        return true;
    }

    feature_value_type operator() (const feature_value_type featureValue) const
    {
        return prior *
                boost::math::pdf(distribution,
                                 (featureValue - mean)/stdDev); //normalize the featurValue before discovering its probability
    }

private:
    boost::math::laplace_distribution<feature_value_type> distribution;
    feature_value_type prior, mean, stdDev;
};



/**
 * Estimates the probability of a certain feature value using a histogram. It is
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
        in >> prior;

        int buckets = 0;
        in >> buckets;

        for (int i = 0; i < buckets; ++i)
        {
            feature_value_type p;
            in >> p;
            histogram.push_back(p);
        }

        return true;
    }

    bool write(std::ostream & out) const
    {
        out << ' ' << prior << ' ' << histogram.size();

        for (unsigned int i = 0; i < histogram.size(); ++i)
        {
            out << ' ' << histogram[i];
        }

        return true;
    }

    feature_value_type operator() (const feature_value_type featureValue) const
    {
        const int buckets = histogram.size();
        const int index = featureValue >= std::sqrt(2) ? buckets :
                          featureValue <= -std::sqrt(2) ? 0 :
                          (int)((buckets/2.0) * featureValue / std::sqrt(2)) + (buckets/2);
        return histogram[index] * prior;
    }

    std::vector<feature_value_type> histogram;
    feature_value_type prior;
};



/**
 * A quadratic discriminant to be used by the feature classifier.
 */
class SingleVariableQuadraticDiscriminant {
public:
    SingleVariableQuadraticDiscriminant() : mean(0), variance(1), prior(1) {}

    SingleVariableQuadraticDiscriminant(feature_value_type variance_,
                                        feature_value_type mean_,
                                        feature_value_type prior_) : mean(mean_),
                                                                     variance(variance_),
                                                                     prior(prior_) {}

    bool read(std::istream & in)
    {
        in >> mean
           >> variance
           >> prior;
        return true;
    }

    bool write(std::ostream & out) const
    {
        out << ' '
            << mean << ' '
            << variance << ' '
            << prior;
        return true;
    }

    feature_value_type operator() (const feature_value_type featureValue) const
    {
        return featureValue * featureValue / (-2.0 * variance) +
               featureValue * mean / variance +
               (mean * mean / (-2.0 * variance) - std::log(variance)/2.0 + std::log(prior));
    }

private:
    feature_value_type mean, variance, prior;
};



/**
 * A classifier that evaluates a single feature using 2 different weights: positive and negative.
 * It compares the positive and negative result and classifies accordingly.
 */
template <typename PositiveProbabilityEvaluatorType, typename NegativeProbabilityEvaluatorType>
class DualWeightVectorBayesWeakClassifier
{
public:

    DualWeightVectorBayesWeakClassifier() {}

    DualWeightVectorBayesWeakClassifier(const DualWeightVectorBayesWeakClassifier & c) : feature(c.feature),
                                                         positiveProbability(c.positiveProbability),
                                                         negativeProbability(c.negativeProbability) {}

    DualWeightVectorBayesWeakClassifier & operator=(const DualWeightVectorBayesWeakClassifier & c)
    {
        feature = c.feature;
        positiveProbability = c.positiveProbability;
        negativeProbability = c.negativeProbability;

        return *this;
    }

    ~DualWeightVectorBayesWeakClassifier() {}

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
        return pos > neg ? yes : no;
    }

private:
    DualWeightHaarWavelet feature;
    IntensityNormalizedWaveletEvaluator evaluator; //TODO use a static variable?

    PositiveProbabilityEvaluatorType positiveProbability;
    NegativeProbabilityEvaluatorType negativeProbability;
};


typedef DualWeightVectorBayesWeakClassifier<NormalFeatureValueProbability, HistogramFeatureValueProbability> NormalAndHistogramHaarClassifier;
typedef DualWeightVectorBayesWeakClassifier<NormalFeatureValueProbability, NormalFeatureValueProbability>    NormalAndNormalHaarClassifier;
typedef DualWeightVectorBayesWeakClassifier<LaplaceFeatureValueProbability, NormalFeatureValueProbability>   LaplaceAndNormalHaarClassifier;

typedef DualWeightVectorBayesWeakClassifier<HistogramFeatureValueProbability, HistogramFeatureValueProbability> HistogramAndHistogramHaarClassifier;



/**
 * A classifier that uses as classification criteria the naive Bayes rule.
 */
template <typename DiscriminantType>
class SingleWeightVectorBayesWeakClassifier
{
public:

    SingleWeightVectorBayesWeakClassifier() {}

    SingleWeightVectorBayesWeakClassifier(const SingleWeightVectorBayesWeakClassifier & c) : feature(c.feature),
                                                         positiveDiscriminant(c.positiveDiscriminant),
                                                         negativeDiscriminant(c.negativeDiscriminant) {}

    SingleWeightVectorBayesWeakClassifier & operator=(const SingleWeightVectorBayesWeakClassifier & c)
    {
        feature = c.feature;
        positiveDiscriminant = c.positiveDiscriminant;
        negativeDiscriminant = c.negativeDiscriminant;

        return *this;
    }

    ~SingleWeightVectorBayesWeakClassifier() {}

    virtual bool read(std::istream & in)
    {
        return feature.read(in)
                && positiveDiscriminant.read(in)
                && negativeDiscriminant.read(in);
    }

    virtual bool write(std::ostream & out) const
    {
        return feature.write(out)
                && positiveDiscriminant.write(out)
                && negativeDiscriminant.write(out);
    }

    //This is supposed to be used only during trainning and ROC curve construction
    float featureValue(const Example &example, const float scale = 1.0f) const
    {
        const float featureValue = evaluator(feature, example.getIntegralSum(), example.getIntegralSquare(), scale);
        return positiveDiscriminant(featureValue) - negativeDiscriminant(featureValue);
    }

    Classification classify(const Example &example, const float scale = 1.0f) const
    {
        const float featureValue = evaluator(feature, example.getIntegralSum(), example.getIntegralSquare(), scale);
        const feature_value_type pos = positiveDiscriminant(featureValue);
        const feature_value_type neg = negativeDiscriminant(featureValue);
        return pos > neg ? yes : no;
    }

private:
    HaarWavelet feature;
    VarianceNormalizedWaveletEvaluator evaluator; //Notice the usage of VARIANCE normalization, instead of intensity normalization

    DiscriminantType positiveDiscriminant, negativeDiscriminant;
};


typedef SingleWeightVectorBayesWeakClassifier<SingleVariableQuadraticDiscriminant> AdhikariHaarClassifier;


#endif // WEAKHYPOTHESIS_h
