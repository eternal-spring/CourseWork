#include<vector>

using matrix = std::vector<std::vector<int>>;

class MatrixCalculator
{
public:
	static matrix Identity(int size);
	static matrix Subtract(const matrix& leftMatrix, const matrix& rightMatrix);
	static matrix Transpose(const matrix& original);
	static matrix Multiply(const matrix& leftMatrix, const matrix& rightMatrix, bool parallel);
};

