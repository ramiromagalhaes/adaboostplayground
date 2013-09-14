#ifndef HAARCLASSIFIER_H
#define HAARCLASSIFIER_H

#include "common.h"
#include "haarwavelet.h"
#include "labeledexample.h"

class HaarClassifier
{
private:
    HaarWavelet * wavelet;
    std::vector<float> mean;
    float stdDev;
    float q;

public:
    //About those constructors and operator=, see the links below:
    //http://pages.cs.wisc.edu/~hasti/cs368/CppTutorial/NOTES/CLASSES-PTRS.html#destructor
    //http://stackoverflow.com/questions/6435404/c-error-double-free-or-corruption-fasttop

    HaarClassifier();

    HaarClassifier(HaarWavelet * w);

    HaarClassifier(const HaarClassifier & c);

    HaarClassifier & operator=(const HaarClassifier & c);

    virtual ~HaarClassifier();

    bool read(std::ifstream & in);

    bool write(std::ofstream & out) const;

    Classification classify(LabeledExample & example) const;

    static bool loadClassifiers(cv::Size * const size, const std::string &filename, std::vector<HaarClassifier> & classifiers);
};

#endif // HAARCLASSIFIER_H
