#ifndef HAARCLASSIFIER_H
#define HAARCLASSIFIER_H

#include "common.h"
#include "haarwavelet.h"
#include "labeledexample.h"

#include <fstream>

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

    bool read(std::istream & in);

    bool write(std::ostream & out) const;

    Classification classify(LabeledExample & example) const;

    static bool loadClassifiers(const std::string &filename, std::vector<HaarClassifier> & classifiers);
};

#endif // HAARCLASSIFIER_H
