/*
 * StrongHypothesis.h
 *
 *  Created on: 06/03/2013
 *      Author: ramiro
 */

#ifndef STRONGHYPOTHESIS_H_
#define STRONGHYPOTHESIS_H_

#include <vector>
#include <utility>
#include "Common.h"
#include "WeakHypothesis.h"

template<typename dataType> class StrongHypothesis {
private:
	std::vector< std::pair<double, WeakHypothesis<dataType> > > hypothesis;

public:
	//note: intentional inline methods thanks to templates. See http://stackoverflow.com/questions/644397/c-class-with-template-cannot-find-its-constructor
	StrongHypothesis() {}
	virtual ~StrongHypothesis() {}

	void include(const double alpha, const WeakHypothesis<dataType> weak_hypothesis) {
		std::pair<double, WeakHypothesis<dataType> > p(alpha, weak_hypothesis);
		hypothesis.insert(hypothesis.end(), p);
	}

	Classification classify(const dataType &input) {
		double result = 0;

		for (typename std::vector< std::pair<double, WeakHypothesis<dataType> > >::iterator it = hypothesis.begin(); it != hypothesis.end(); ++it) {
			std::pair<double, WeakHypothesis<dataType> > wh = *it;
			result += (wh.first) * (wh.second.test(input));
		}

		return result >= 0 ? yes : no;
	}

};

#endif /* STRONGHYPOTHESIS_H_ */
