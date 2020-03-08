#include <iostream>
#include <Eigen/Dense>
#include "file_manage.h"
#include "decision_tree.h"
#include "model_selection.h"
using namespace std;
using namespace Eigen;

int main()
{
	MatrixXd features;
	VectorXd labels;

	read_csv("data/iris.csv", features, labels, 150, 5);
	//read_csv("data/seeds.csv", features, labels, 210, 8);

	DecisionTree model;
	model.fit(features, labels);
	model.printTree();

	double accuracy = evaluate_model(model, features, labels, 5);
	cout << "Model accuracy : " << accuracy << endl;

	return 0;
}