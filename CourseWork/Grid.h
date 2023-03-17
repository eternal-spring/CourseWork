#include<vector>
#pragma once
using flow = std::vector<double>;

class Grid
{
private:
	flow mValues;

public:
	Grid(flow values);

	void SetValues(flow values);

	flow getValues() { return mValues; }
};

