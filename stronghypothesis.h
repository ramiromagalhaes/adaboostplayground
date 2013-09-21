#ifndef STRONGHYPOTHESIS_H_
#define STRONGHYPOTHESIS_H_

#include <vector>
#include <string>
#include <fstream>

#include "common.h"
#include "labeledexample.h"

/**
 * Instances of this class hold the weights and weak classifiers that make the strong classifier.
 * Notice that this class holds pointers to the weak classifiers, added via the insert method.
 * All those pointers will be deleted on destruction of this class.
 */
template<typename WeakHypothesisType>
class StrongHypothesis {
private:

    /**
     * @brief The entry class holds the WeakHypothesis pointer and it's weight.
     */
    class entry {
    public:
        weight_type alpha;
        WeakHypothesisType weakHypothesis;

        entry() : alpha(0),
                  weakHypothesis() {}

        entry(weight_type a, WeakHypothesisType h) : alpha(a),
                                                     weakHypothesis(h) {}
    };

    float threshold;
    std::vector<entry> hypothesis;

    const bool writeToStreamOnInsert;
    const std::string path;

public:
    StrongHypothesis() : threshold(0),
                         hypothesis(0),
                         writeToStreamOnInsert(false) {}

    StrongHypothesis(weight_type threshold_) : threshold(threshold_),
                                               hypothesis(0),
                                               writeToStreamOnInsert(false) {}

    /**
     * This constructor should never be used, except during trainning.
     */
    StrongHypothesis(std::string path_) : threshold(0),
                                          hypothesis(0),
                                          writeToStreamOnInsert(true),
                                          path(path_)
    {
        std::ofstream out( path.c_str(), std::ios::trunc );
        if ( !out.is_open() )
        {
            throw 219;
        }
        out.close();
    }

    ~StrongHypothesis()
    {
        hypothesis.clear(); //each entry object destroys its contents
    }



    void insert(weight_type alpha, WeakHypothesisType weak_hypothesis) {
        hypothesis.push_back( entry(alpha, weak_hypothesis) );

        if (writeToStreamOnInsert)
        {
            std::ofstream out( path.c_str(), std::ios::trunc );
            if ( !out.is_open() )
            {
                throw 220;
            }
            write(out);
            out.close();
        }
    }



    Classification classify(const LabeledExample & sample) const {
        weight_type result = .0f;

        for (typename std::vector<entry>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it) {
            result += (it->alpha) * (it->weakHypothesis.classify(sample));
        }

        return result >= threshold ? yes : no;
    }



    weight_type classificationValue(const LabeledExample & sample) const {
        weight_type result = .0f;

        for (typename std::vector<entry>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it) {
            result += (it->alpha) * (it->weakHypothesis.classify(sample));
        }

        return result;
    }



    bool write(std::ostream & out)
    {
        for(typename std::vector<entry>::iterator it = hypothesis.begin(); it != hypothesis.end(); ++it)
        {
            out << it->alpha << ' ';
            if ( !it->weakHypothesis.write(out) )
            {
                return false;
            }
            out << std::endl;
        }

        return true;
    }



    bool read(std::istream & in)
    {
        while( true )
        {
            weight_type a;
            in >> a;

            WeakHypothesisType weak_hypothesis;
            if ( !weak_hypothesis.read(in) )
            {
                return false;
            }

            if ( in.eof() )
            {
                break;
            }

            insert(a, weak_hypothesis);
        }

        return true;
    }

    //TODO accessor methods to the weak hypothesis
};



#endif /* STRONGHYPOTHESIS_H_ */
