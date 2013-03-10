/*
 * WeakHypothesis.h
 *
 *  Created on: 06/03/2013
 *      Author: ramiro
 */

#ifndef WEAKHYPOTHESIS_H_
#define WEAKHYPOTHESIS_H_

#include "Classification.h"

template<typename dataType> class WeakHypothesis {
public:
	WeakHypothesis() { }
	virtual ~WeakHypothesis() { }

	Classification test(const dataType &data) {
		return no;
	}

};

#endif /* WEAKHYPOTHESIS_H_ */
