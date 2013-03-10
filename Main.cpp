/*
 * Main.cpp
 *
 *  Created on: 02/03/2013
 *      Author: ramiro
 */



#include <vector>
#include "Common.h"
#include "WeakLearner.h"
#include "StrongHypothesis.h"
#include "Adaboost.h"

#include <iostream>



class Point {
public:
	double x, y;

	Point () {
		x = y = 0;
	}

	Point(double x, double y) {
		this->x = x;
		this->y = y;
	}

	virtual ~Point() {}
};



enum Orientation {
	vertical, horizontal
};



class MyWeakHypothesis : public WeakHypothesis<Point> {
public:
	Orientation orientation;
	double position;

	MyWeakHypothesis(Orientation o, double p) : WeakHypothesis() {
		orientation = o;
		position = p;
	}

	virtual Classification test(const Point &input) {
		switch (orientation) {
			case vertical:
				//if input is to the left of the vertical bar, then return yes
				if (input.x <= position) {
					return yes;
				} else {
					return no;
				}
			case horizontal:
				//if input is above the horizontal bar, then return yes
				if (input.y >= position) {
					return yes;
				} else {
					return no;
				}
			default:
				throw "!!!";
		}
	}

};



class MyWeakLearner : public WeakLearner<Point> {
private:
	std::vector<MyWeakHypothesis> hypothesis;

public:
	MyWeakLearner() : WeakLearner<Point>() {
		MyWeakHypothesis *h = new MyWeakHypothesis(vertical, 3);
		hypothesis.insert(hypothesis.end(), *h);

		h = new MyWeakHypothesis(vertical, 7);
		hypothesis.insert(hypothesis.end(), *h);

		h = new MyWeakHypothesis(horizontal, 2);
		hypothesis.insert(hypothesis.end(), *h);

		h = new MyWeakHypothesis(horizontal, 6);
		hypothesis.insert(hypothesis.end(), *h);
	};

	~MyWeakLearner(){};

	virtual WeakHypothesis<Point> learn(const std::vector < training_data<Point> > &data) {
		double lowest_error = 1;
		std::vector < training_data<Point> >::size_type best_hypothesis_index = 0;

		for (std::vector < training_data<Point> >::size_type i = 0; i < hypothesis.size(); i++) {
			MyWeakHypothesis hyp = hypothesis[i];
			int wrong_conclusions = 0;

			for (typename std::vector < training_data<Point> >::const_iterator itTrain = data.begin(); itTrain != data.end(); ++itTrain) {
				training_data<Point> train = *itTrain;

				if (hyp.test(train.data) != train.classification) {
					wrong_conclusions++;
				}
			}

			const double error_ratio = (double)wrong_conclusions / (double)data.size();

			if (error_ratio < lowest_error) {
				lowest_error = error_ratio;
				best_hypothesis_index = i;
			}
		}

		return hypothesis[best_hypothesis_index];
	}
};



std::vector < training_data <Point> > get_training_data() {
	std::vector < training_data <Point> > data;

	Point *p = new Point(1, 2);
	training_data<Point> *d = new training_data<Point>(*p, yes);
	data.insert(data.end(), *d);

	p = new Point(2, 7);
	d = new training_data<Point>(*p, yes);
	data.insert(data.end(), *d);

	p = new Point(3, 9);
	d = new training_data<Point>(*p, yes);
	data.insert(data.end(), *d);

	p = new Point(4, 4);
	d = new training_data<Point>(*p, yes);
	data.insert(data.end(), *d);

	p = new Point(5, 6);
	d = new training_data<Point>(*p, yes);
	data.insert(data.end(), *d);

	p = new Point(6, 2);
	d = new training_data<Point>(*p, yes);
	data.insert(data.end(), *d);

	p = new Point(7, 9);
	d = new training_data<Point>(*p, no);
	data.insert(data.end(), *d);

	p = new Point(8, 5);
	d = new training_data<Point>(*p, no);
	data.insert(data.end(), *d);

	p = new Point(9, 7);
	d = new training_data<Point>(*p, no);
	data.insert(data.end(), *d);

	p = new Point(8, 3);
	d = new training_data<Point>(*p, no);
	data.insert(data.end(), *d);

	return data;
}



int main(int argc, char **argv) {
	std::vector < training_data <Point> > training_data = get_training_data();

	WeakLearner<Point> *learner = new MyWeakLearner();
	StrongHypothesis<Point> strong_hypothesis;
	Adaboost<Point> boosting(learner);
	boosting.train(training_data, strong_hypothesis);

	{
		Point p(2, 8);
		std::cout << strong_hypothesis.classify(p) << std::endl;
	}
	{
		Point p(6, 6);
		std::cout << strong_hypothesis.classify(p) << std::endl;
	}
	{
		Point p(1, 1);
		std::cout << strong_hypothesis.classify(p) << std::endl;
	}
	{
		Point p(7, 9);
		std::cout << strong_hypothesis.classify(p) << std::endl;
	}
	{
		Point p(8, 3);
		std::cout << strong_hypothesis.classify(p) << std::endl;
	}

	return 0;
}
