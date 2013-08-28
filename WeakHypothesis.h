#ifndef WEAKHYPOTHESIS_H_
#define WEAKHYPOTHESIS_H_

#include <ostream>
#include <string>

#include "Common.h"

template<typename dataType> class WeakHypothesis {
public:
	WeakHypothesis() { }
	virtual ~WeakHypothesis() { }

	virtual Classification classify(const dataType &data) const =0;

	virtual std::string str() const =0;

};

#endif /* WEAKHYPOTHESIS_H_ */
