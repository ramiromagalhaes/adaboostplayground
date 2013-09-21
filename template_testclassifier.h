#ifndef TEMPLATE_TESTCLASSIFIER_H
#define TEMPLATE_TESTCLASSIFIER_H

#include <vector>
#include <iostream>

#include <tbb/tbb.h>
#include <opencv2/core/core.hpp>

#include "common.h"
#include "stronghypothesis.h"
#include "haarclassifier.h"
#include "sampleextractor.h"



struct RocRecord
{
    int falsePositives;
    int truePositives;
    float value;

    RocRecord() : falsePositives(0), truePositives(0), value(0) {}

    bool operator < (const RocRecord & rh) const
    {
        return falsePositives < rh.falsePositives;
    }
};



template<typename WeakHypothesisType>
struct RoiCalculator
{
    std::vector<LabeledExample> const * const samples;
    std::vector<RocRecord> * const records;
    StrongHypothesis<WeakHypothesisType> * const strongHypothesis;

    RoiCalculator(std::vector<LabeledExample> const * const samples_,
                  std::vector<RocRecord> * records_,
                  StrongHypothesis<WeakHypothesisType> * const strongHypothesis_) : samples(samples_),
                                                                                    records(records_),
                                                                                    strongHypothesis(strongHypothesis_) {}

    void operator()(tbb::blocked_range< unsigned int > & range) const
    {
        for (unsigned int i = range.begin(); i < range.end(); ++i)
        {
            (*records)[i].value = strongHypothesis->classificationValue( (*samples)[i] );

            for (unsigned int j = 0; j < samples->size(); ++j)
            {
                const float classVal = strongHypothesis->classificationValue( (*samples)[j] );
                (*records)[i].truePositives  += (classVal >= (*records)[i].value) && ((*samples)[j].getLabel() == yes);
                (*records)[i].falsePositives += (classVal >= (*records)[i].value) && ((*samples)[j].getLabel() == no);
            }
        }
    }
};



template<typename WeakHypothesisType>
int ___main(const std::string positivesFile,
            const std::string negativesFile,
            const std::string strongHypothesisFile)
{
    StrongHypothesis<WeakHypothesisType> strongHypothesis;
    {
        std::ifstream in(strongHypothesisFile.c_str());
        if ( !in.is_open() )
        {
            return 7;
        }
        if ( !strongHypothesis.read(in) )
        {
            return 11;
        }
    }


    std::vector<LabeledExample> samples;
    {
        std::vector<LabeledExample> positiveSamples, negativeSamples;

        if ( !SampleExtractor::fromImageFile(positivesFile, positiveSamples, yes) )
        {
            return 13;
        }
        std::cout << "Loaded " << positiveSamples.size() << " positive samples." << std::endl;

        if ( !SampleExtractor::extractRandomSample(10000, negativesFile, negativeSamples, no) )
        {
            return 17;
        }
        std::cout << "Loaded " << negativeSamples.size() << " negative samples." << std::endl;

        samples.reserve( positiveSamples.size() + negativeSamples.size() ); //reserve, not resize
        samples.insert( samples.end(), positiveSamples.begin(), positiveSamples.end() );
        samples.insert( samples.end(), negativeSamples.begin(), negativeSamples.end() );
    }

    std::vector<RocRecord> records(samples.size());
    tbb::parallel_for( tbb::blocked_range< unsigned int >(0, samples.size()),
                       RoiCalculator<WeakHypothesisType>(&samples, &records, &strongHypothesis) );

    std::sort(records.begin(), records.end());

    for (unsigned int i = 0; i < samples.size(); ++i)
    {
        std::cout << records[i].falsePositives << ' ' << records[i].truePositives << std::endl;
    }

    return 0;
}

#endif // TEMPLATE_TESTCLASSIFIER_H
