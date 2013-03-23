/*
 * Main.cpp
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */



#include <limits>
#include <vector>
#include <iostream>
#include <sstream> //for std::stringstream
#include <string>
#include <ostream>

#include "Common.h"
#include "WeakLearner.h"
#include "StrongHypothesis.h"
#include "Adaboost.h"



class Point {
public:
	double x, y;

	Point () : x(0), y(0) {}
	Point(double _x, double _y) : x(_x), y(_y) {}
	virtual ~Point() {}

	friend std::ostream& operator<<(std::ostream& os, Point& s) {
		os << "Point(" << s.x << ", " << s.y << ")";
		return os;
	}

};



class MyWeakHypothesis : public WeakHypothesis<Point> {
public:
	enum Orientation {
		vertical = -1, horizontal = 1
	};

	enum Inclusion {
		before = -1, after = 1
	};

	Orientation orientation;
	Inclusion inclusion;
	double position;

	MyWeakHypothesis(Orientation o, Inclusion i, double p) : WeakHypothesis<Point>() {
		//orientation(o), inclusion(i), position(p)
		orientation = o;
		inclusion = i;
		position = p;
	}

	virtual Classification classify(const Point &input) const {
		switch (orientation) {
			case vertical  : return inclusion * (input.x - position) >= 0 ? yes : no;
			case horizontal: return inclusion * (input.y - position) >= 0 ? yes : no;
			default        : throw 1;
		}
	}

	virtual std::string str() const {
		std::string os("Hypothesis ");
		switch (orientation) {
			case vertical  : os.append("V "); break;
			case horizontal: os.append("H "); break;
			default        : throw 3;
		}

		switch (inclusion) {
			case before : os.append("B "); break;
			case after  : os.append("A "); break;
			default        : throw 4;
		}

		{
			std::ostringstream oss;
			oss << position;
			os.append(oss.str());
		}

		os.append("; ");
		return os;
	}
};



class MyWeakLearner : public WeakLearner<Point> {
private:
	std::vector<MyWeakHypothesis*> hypothesis;

public:
	MyWeakLearner() : WeakLearner<Point>() {
		for (int i = 1; i < 10; i++) {
			MyWeakHypothesis *h = new MyWeakHypothesis(MyWeakHypothesis::vertical, MyWeakHypothesis::before, i);
			hypothesis.insert(hypothesis.end(), h);

			h = new MyWeakHypothesis(MyWeakHypothesis::vertical, MyWeakHypothesis::after, i);
			hypothesis.insert(hypothesis.end(), h);

			h = new MyWeakHypothesis(MyWeakHypothesis::horizontal, MyWeakHypothesis::before, i);
			hypothesis.insert(hypothesis.end(), h);

			h = new MyWeakHypothesis(MyWeakHypothesis::horizontal, MyWeakHypothesis::after, i);
			hypothesis.insert(hypothesis.end(), h);
		}
	};

	~MyWeakLearner(){};

	virtual WeakHypothesis<Point>* learn(
			const std::vector < LabeledExample<Point> * > &training_set,
			const std::vector < double > &weight_distribution,
			double &weighted_error) {

		/*
		const int m = training_set.size(); //the size of sample we'll take from the trainingData to train a WeakLearner each round

		//get a sample of the trainingData...
		std::vector < LabeledExample <Point> * > training_sample(m);
		std::vector < double > training_sample_weight_distribution(m);
		resample(m, weight_distribution, training_set, training_sample, training_sample_weight_distribution);
		*/

		double lowest_error = std::numeric_limits<double>::max();
		std::vector < MyWeakHypothesis* >::size_type best_hypothesis_index = 0;

		for (std::vector < MyWeakHypothesis* >::size_type i = 0; i < hypothesis.size(); i++) {
			MyWeakHypothesis const * const hyp = hypothesis[i];

			double weak_hypothesis_weight = 0;

			for (std::vector < LabeledExample<Point> * >::size_type j = 0; j < training_set.size(); j++) {
				if (hyp->classify(training_set[j]->example) != training_set[j]->label) {
					weak_hypothesis_weight += weight_distribution[j];
				}
			}

			if (weak_hypothesis_weight < lowest_error) {
				lowest_error = weak_hypothesis_weight;
				best_hypothesis_index = i;
			}

			//std::cout << "Erro de " << i << ": " << ((double)error_counter / (double)training_set.size()) << std::endl;
		}

		weighted_error = lowest_error;


		return hypothesis[best_hypothesis_index];
	}
};



std::vector < LabeledExample <Point> * >* get_training_data() {
	std::vector < LabeledExample <Point> * > * const data = new std::vector < LabeledExample <Point> * >();

	Point *p;
	LabeledExample<Point> * d;

	p = new Point(3, 5);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(4, 4);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(4, 5);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(5, 4);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(5, 5);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(6, 5);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(6, 6);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(7, 4);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(7, 5);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(8, 4);
	d = new LabeledExample<Point>(*p, yes);
	data->push_back(d);

	p = new Point(2, 4);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(2, 5);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(3, 2);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(3, 3);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(3, 4);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(3, 6);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(4, 1);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(4, 2);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(4, 3);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(4, 6);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(4, 7);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(5, 2);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(5, 3);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(5, 6);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(5, 7);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(6, 2);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(6, 3);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(6, 4);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(6, 7);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(7, 2);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(7, 3);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(7, 6);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(7, 7);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(8, 2);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(8, 3);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(8, 5);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(8, 6);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(9, 4);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	p = new Point(9, 5);
	d = new LabeledExample<Point>(*p, no);
	data->push_back(d);

	return data;
}

int main(int argc, char **argv) {
	try {
		std::vector < LabeledExample <Point> * > const * const training_set = get_training_data();

		WeakLearner<Point> *learner = new MyWeakLearner();
		StrongHypothesis<Point> strong_hypothesis;
		Adaboost<Point> boosting(learner);

		boosting.train(*training_set, strong_hypothesis);

		std::cout << strong_hypothesis;
		std::cout << std::endl;

		for (std::vector < LabeledExample <Point> * >::const_iterator it = training_set->begin(); it != training_set->end(); ++it) {
			LabeledExample<Point> * const le = *it;
			std::cout << le->example << " esperado " << le->label << " obtido " << strong_hypothesis.classify(le->example) << std::endl;
		}
	} catch (int e) {
		std::cout << "Erro " << e << std::endl;
	}

	return 0;
}
