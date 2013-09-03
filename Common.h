#ifndef CLASSIFICATION_H_
#define CLASSIFICATION_H_



/**
 * @brief weight_type Weighs used for training should adopt this type.
 *                    We use float as a default.
 */
typedef float weight_type;



/**
 * @brief The Classification enum holds default values for the binary classification case.
 */
enum Classification {
    no = -1, yes = 1
};



/**
 * Holds the sample and its classification.
 */
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
