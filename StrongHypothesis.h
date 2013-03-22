/*
 * StrongHypothesis.h
 *
 *  Created on: 06/03/2013
 *      Author: ramiro
 */

#ifndef STRONGHYPOTHESIS_H_
#define STRONGHYPOTHESIS_H_

#include <vector>
#include <ostream>
#include <iostream>
#include <iomanip>
#include "Common.h"
#include "WeakHypothesis.h"


/*
 * Instances of this class hold the weights and weak classifiers that make the strong classifier.
 */
template<typename dataType> class StrongHypothesis {
private:

	class entry {
	public:
		double weight;
		WeakHypothesis<dataType> * weakHypothesis;

		entry(double w, WeakHypothesis<dataType> * h) : weight(w), weakHypothesis(h) {}
		virtual ~entry() {}
	};

	std::vector<entry> hypothesis;

public:
	//note: intentional inline methods thanks to templates. See http://stackoverflow.com/questions/644397/c-class-with-template-cannot-find-its-constructor
	StrongHypothesis() {}
	virtual ~StrongHypothesis() {
		//We won't destroy all elements inside the hypothesis because we expect the weak learner to do that.
	}



	void insert(double alpha, WeakHypothesis<dataType> * weak_hypothesis) {
		entry e(alpha, weak_hypothesis);
		hypothesis.insert(hypothesis.end(), e);
	}



	Classification classify(const dataType &input) const {
		double result = 0;

		for (typename std::vector<entry>::const_iterator it = hypothesis.begin(); it != hypothesis.end(); ++it) {
			entry e = *it;
			result += (e.weight) * (e.weakHypothesis->classify(input));
		}

		return result >= 0 ? yes : no;
	}



	//TODO lean more about C++ friend operator. It is a funny thing...
	friend std::ostream& operator<<(std::ostream& os, StrongHypothesis<dataType>& s) {
		os << "Strong Hypothesis {" << std::endl;
		for (typename std::vector<entry>::const_iterator it = s.hypothesis.begin(); it != s.hypothesis.end(); ++it) {
			os << "\t " << std::fixed << std::setprecision(4) << (*it).weight << ' ';

			WeakHypothesis<dataType> const * const wht = (*it).weakHypothesis;

			os << wht->str();
			os << std::endl;
		}

		os << '}' << std::endl;

		return os;
	}
};



#endif /* STRONGHYPOTHESIS_H_ */
