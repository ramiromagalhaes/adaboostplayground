/*
 * Classification.h
 *
 *  Created on: 06/03/2013
 *      Author: ramiro
 */

#ifndef CLASSIFICATION_H_
#define CLASSIFICATION_H_



enum Classification {
	no = -1, yes = 1
};

//"typedef"
template < typename dataType > class training_data {
public:
	dataType data;
	Classification classification;

	training_data() {
		classification = no;
	}

	training_data (const dataType d, const Classification c) : data(d), classification(c) {
	}
};

#endif /* CLASSIFICATION_H_ */
