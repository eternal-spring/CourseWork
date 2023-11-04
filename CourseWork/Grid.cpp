#include "Grid.h"

Grid::Grid(std::vector<double> values) {
	SetValues(values);
}

void Grid::SetValues(std::vector<double> values) {
	mValues = values;
}