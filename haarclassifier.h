#ifndef HAARCLASSIFIER_H
#define HAARCLASSIFIER_H

#include <fstream>

#include <opencv2/core/core.hpp>

#include "common.h"
#include "haarwavelet.h"
#include "labeledexample.h"

class HaarClassifier
{
private:
    HaarWavelet * wavelet;
    std::vector<float> mean;
    float stdDev; //TODO Not used. Will soon drop it.
    float q;
    float p;

    Classification do_classify() const;

public:
    //About those constructors and operator=, see the links below:
    //http://pages.cs.wisc.edu/~hasti/cs368/CppTutorial/NOTES/CLASSES-PTRS.html#destructor
    //http://stackoverflow.com/questions/6435404/c-error-double-free-or-corruption-fasttop

    HaarClassifier();

    HaarClassifier(HaarWavelet * w);

    HaarClassifier(const HaarClassifier & c);

    HaarClassifier & operator=(const HaarClassifier & c);

    virtual ~HaarClassifier();

    bool read(std::istream & in);

    bool write(std::ostream & out) const;

    void setThreshold(const float q_);

    void setPolarity(const float p_);

    float featureValue(LabeledExample & example) const;

    Classification classify(LabeledExample & example) const;

    Classification classify(cv::Mat & example) const;

    static bool loadClassifiers(const std::string &filename, std::vector<HaarClassifier> & classifiers);
};

#endif // HAARCLASSIFIER_H
