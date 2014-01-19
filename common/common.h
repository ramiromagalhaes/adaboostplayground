#ifndef COMMON_H_
#define COMMON_H_

#include <vector>



/**
 * @brief weight_type Weighs used for training should adopt this type.
 *                    We use float as a default.
 */
typedef float weight_type; // TODO move to haarcommon



/**
 * @brief weight_type Holds feature values.
 */
typedef float feature_value_type; //TODO move to haarcommon



/**
 * @brief WeightVector A vector to hold instances of weight_type.
 */
typedef std::vector<weight_type> WeightVector;



/**
 * @brief The Classification enum holds default values for the binary classification case.
 */
enum Classification {
    no = -1, yes = 1
};



#endif /* COMMON_H_ */
