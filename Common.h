#ifndef CLASSIFICATION_H_
#define CLASSIFICATION_H_



enum Classification {
	no = -1, yes = 1
};

//"typedef"
template < typename dataType > class LabeledExample {
public:
	dataType example;
	Classification label;

	LabeledExample() {
		label = no;
	}

	LabeledExample (const dataType d, const Classification c) : example(d), label(c) {
	}
};

#endif /* CLASSIFICATION_H_ */
