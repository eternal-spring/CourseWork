#include "pch.h"
#include <Decomposition.cpp>
#include <Grid.cpp>

class DecompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        original = { 0, 1, 2, 3, 4, 5, 6 };
        enlarged = { 0, 2, 5, 6 };
        embedding = { { 1, 1, 0, 0, 0, 0 }, { 0, 0, 1, 1, 1, 0 }, { 0, 0, 0, 0, 0, 1 } };
        extension = { { 1, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1 } };
        main = { 0, 2, 5 };
        wavelet = { 0,1,0,1,2,0 };
    }
    flow original;
    flow enlarged;
    matrix embedding;
    matrix extension;
    flow main;
    flow wavelet;
};

TEST_F(DecompositionTest, DecompositionWorks) {
    Decomposition decomposition(Grid(original), Grid(enlarged), false);
    ASSERT_EQ(decomposition.getMainFlow(), main);
    EXPECT_EQ(decomposition.getWaveletFlow(), wavelet);
    EXPECT_EQ(decomposition.getEmbeddingMatrix(), embedding);
    EXPECT_EQ(decomposition.getExtensionMatrix(), extension);
    Decomposition parallel_decomposition(Grid(original), Grid(enlarged), true);
    ASSERT_EQ(parallel_decomposition.getMainFlow(), main);
}