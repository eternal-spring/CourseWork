#include "pch.h"
#include <MatrixCalculator.cpp>

class MatrixCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        left = { {1,2,3} , {4,5,6},{7,8,9} };
        right = { {0, 1, 2}, {2, 3, 4}, {4, 5, 6} };
        difference = { {1, 1, 1}, {2, 2, 2}, {3, 3, 3} };
        product = { {16, 22, 28}, {34, 49, 64}, {52, 76, 100} };
    }

	matrix left;
	matrix right;
    matrix difference;
    matrix product;
};


TEST_F(MatrixCalculatorTest, SubtractionWorks) {
  ASSERT_EQ(MatrixCalculator::Subtract(left, right), difference);
  EXPECT_NE(MatrixCalculator::Subtract(left, right), product);
}

TEST_F(MatrixCalculatorTest, MultiplicationWorks) {
    ASSERT_EQ(MatrixCalculator::Multiply(left, right, false), product);
    EXPECT_NE(MatrixCalculator::Multiply(left, right, false), difference);
    EXPECT_EQ(MatrixCalculator::Multiply(left, right, true), product);
}
