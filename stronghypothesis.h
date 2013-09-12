#ifndef STRONGHYPOTHESIS_H_
#define STRONGHYPOTHESIS_H_

#include <vector>
#include <string>
#include <fstream>

#include "common.h"



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
        weight_type weight;
        WeakHypothesisType weakHypothesis;

        entry() : weight(0), weakHypothesis(0) {}
        entry(weight_type w, WeakHypothesisType h)
        {
            weight = w;
            weakHypothesis = h;
        }
    };

    std::vector<entry> hypothesis;

    bool writeToStreamOnInsert;
    std::ofstream data; //this will be kept open during this object's whole life.

public:
    //note: intentional inline methods thanks to templates. See http://stackoverflow.com/questions/644397/c-class-with-template-cannot-find-its-constructor
    StrongHypothesis() {
        writeToStreamOnInsert = false;
    }



    StrongHypothesis(std::string path) {
        data.open( path.c_str(), std::ios::trunc );
        if (!data.is_open())
        {
            throw 220;
        }
        writeToStreamOnInsert = true;
    }



    ~StrongHypothesis() {
        hypothesis.clear(); //each entry object destroys its contents
        if (data.is_open())
        {
            data.close();
        }
    }



    void insert(weight_type alpha, WeakHypothesisType weak_hypothesis) {
        if (writeToStreamOnInsert)
        {
            data << alpha << " ";
            weak_hypothesis.write(data);
            data << std::endl;
        }

        hypothesis.push_back( entry(alpha, weak_hypothesis) );
    }



    Classification classify(const cv::Mat &input) const {
        weight_type result = .0f;

        for (typename std::vector<entry>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it) {
            entry e = *it;
            result += (e.weight) * (e.weakHypothesis->classify(input));
        }

        return result >= 0 ? yes : no;
    }



    //TODO accessor methods to the weak hypothesis
};



#endif /* STRONGHYPOTHESIS_H_ */
