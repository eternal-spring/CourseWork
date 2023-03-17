#include "MatrixCalculator.h"
#include <omp.h>

matrix MatrixCalculator::Identity(int size) {
    matrix identity(size, std::vector<int>(size));
    for (size_t i = 0; i < identity.size(); ++i) {
        for (size_t j = 0; j < identity.at(0).size(); ++j) {
            if (i == j) {
                identity.at(i).at(j) = 1;
            }
        }
    }
    return identity;

}

matrix MatrixCalculator::Subtract(const matrix& leftMatrix, const matrix& rightMatrix)
{
    matrix result(leftMatrix.size(), std::vector<int>(leftMatrix.at(0).size()));

    for (size_t i = 0; i < leftMatrix.size(); ++i) {
        for (size_t j = 0; j < leftMatrix.at(0).size(); ++j) {
            result.at(i).at(j) = leftMatrix.at(i).at(j) - rightMatrix.at(i).at(j);
        }
    }
    return result;
}

matrix MatrixCalculator::Transpose(const matrix& original)
{
    matrix transposed(original.at(0).size(), std::vector<int>(original.size()));

    for (size_t i = 0; i < original.size(); ++i) {
        for (size_t j = 0; j < original.at(0).size(); ++j) {
            transposed.at(j).at(i) = original.at(i).at(j);
        }
    }
    return transposed;
}

matrix MatrixCalculator::Multiply(const matrix& leftMatrix, const matrix& rightMatrix, bool parallel)
{
    matrix result(leftMatrix.size(), std::vector<int>(rightMatrix.at(0).size()));
#pragma omp parallel for if (parallel)
    for (int row = 0; row < result.size(); ++row) {
        for (int col = 0; col < result.at(0).size(); ++col) {
            for (int inner = 0; inner < rightMatrix.size(); ++inner) {
                result.at(row).at(col) += leftMatrix.at(row).at(inner) * rightMatrix.at(inner).at(col);
            }
        }
    }
    
    return result;
}