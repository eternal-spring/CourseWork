#include "pch.h"
#include <Decomposition.cpp>
#include <Grid.cpp>

class DecompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        original = { 0, 1, 2, 3, 4, 5, 6 };
        enlarged = { 0, 2, 5, 6 };
    }
    std::vector<double> original;
    std::vector<double> enlarged;
    matrix embedding{ { 1, 1, 0, 0, 0, 0 }, { 0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 0, 0, 1 } };
    matrix extension{ { 1, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1 } };
    flow main{ { 0, 2, 5 } };
    flow wavelet{ {0, 1, 0, 1, 2, 0 } };
};

TEST_F(DecompositionTest, DecompositionWorks) {
    Decomposition decomposition(Grid(original), Grid(enlarged), false);
    ASSERT_EQ(decomposition.getMainFlow(), main);
    EXPECT_EQ(decomposition.getWaveletFlow(), wavelet);
    EXPECT_EQ(decomposition.getEmbeddingMatrix(), embedding);
    //EXPECT_EQ(decomposition.getExtensionMatrix(), extension);
    Decomposition parallel_decomposition(Grid(original), Grid(enlarged), true);
    ASSERT_EQ(parallel_decomposition.getMainFlow(), main);
}