#include <vector>
#include <Eigen/Dense>
#pragma once

using namespace Eigen;

class Grid
{
private:
	std::vector<double> mValues;

public:
	Grid(std::vector<double> values);

	void SetValues(std::vector<double> values);

	VectorXd getValues() { return Map<VectorXd, Unaligned>(mValues.data(), mValues.size()); }

	Grid() = default;
};

