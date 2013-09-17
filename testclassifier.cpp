#include <vector>
#include <iostream>

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
};

bool rocRecordCompare(const RocRecord & lh, const RocRecord & rh)
{
    return lh.falsePositives > rh.falsePositives;
    //return lh.value > rh.value;
}



/**
 * Arguments:
 *     positivesIndexFile
 *     positivesImageFile
 *     negativesIndexFile
 *     negativesImageFile
 *     strongHypothesisInputFile
 */
int main(int, char **argv) {
    const std::string positivesFile = argv[1];
    const std::string negativesFile = argv[2];
    const std::string strongHypothesisFile = argv[3];

    StrongHypothesis strongHypothesis;
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

        if ( !SampleExtractor::fromIndexFile(positivesFile, positiveSamples, yes) )
        {
            return 13;
        }
        std::cout << "Loaded " << positiveSamples.size() << " positive samples." << std::endl;

        if ( !SampleExtractor::extractRandomSample(10000, negativesFile, negativeSamples, no) )
        {
            return 17;
        }
        std::cout << "Loaded " << negativeSamples.size() << " negative samples." << std::endl;

        samples.reserve( positiveSamples.size() + negativeSamples.size() );
        samples.insert( samples.end(), positiveSamples.begin(), positiveSamples.end() );
        samples.insert( samples.end(), negativeSamples.begin(), negativeSamples.end() );
    }


    std::vector<RocRecord> records(samples.size());
    for (unsigned int i = 0; i < samples.size(); ++i)
    {
        records[i].value = strongHypothesis.classificationValue(samples[i]);

        for (unsigned int j = 0; j < samples.size(); ++j)
        {
            float classVal = strongHypothesis.classificationValue(samples[j]);
            records[i].truePositives  += (classVal >= records[i].value) && (samples[j].label == yes);
            records[i].falsePositives += (classVal >= records[i].value) && (samples[j].label == no);
        }
    }
    std::sort(records.begin(), records.end(), rocRecordCompare);

    for (unsigned int i = 0; i < samples.size(); ++i)
    {
        std::cout << records[i].falsePositives << ',' << records[i].truePositives << std::endl;
    }

    return 0;
}
