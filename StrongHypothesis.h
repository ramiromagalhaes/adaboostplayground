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
#include "WeakHypothesis.h"

template<typename dataType> class StrongHypothesis {
private:
	std::vector< std::pair<double, WeakHypothesis<dataType> > > hypothesis;

public:
	StrongHypothesis() {} //note: intentional inline con/destructor. See http://stackoverflow.com/questions/644397/c-class-with-template-cannot-find-its-constructor
	virtual ~StrongHypothesis() {}

	void include(const double alpha, const WeakHypothesis<dataType> weak_hypothesis) {
		std::pair<double, WeakHypothesis<dataType> > p(alpha, weak_hypothesis);
		hypothesis.insert(hypothesis.end(), p);
	}
};

#endif /* STRONGHYPOTHESIS_H_ */
