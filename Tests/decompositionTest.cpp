#include "pch.h"
#include <Decomposition.cpp>
#include <Grid.cpp>
#include "mpi.h"
#include "Eigen/src/Core/Matrix.h"
#include <MpiHelper.cpp>

class MPIEnvironment : public ::testing::Environment
{
public:
    virtual void SetUp() {
        char** argv;
        int argc = 0;
        int mpiError = MPI_Init(&argc, &argv);
        ASSERT_FALSE(mpiError);
    }
    virtual void TearDown() {
        int mpiError = MPI_Finalize();
        ASSERT_FALSE(mpiError);
    }
    virtual ~MPIEnvironment() {}
};

class DecompositionTest : public ::testing::Test {
protected:
    void SetUp() override {
        original = { 0, 1, 2, 3, 4, 5, 6 };
        enlarged = { 0, 2, 5, 6 };
    }
    std::vector<double> original;
    std::vector<double> enlarged;
    VectorXd main{ { 0, 2, 5 } };
    VectorXd wavelet{ {0, 1, 0, 1, 2, 0 } };
};

TEST_F(DecompositionTest, DecompositionWorks) {
    int numtasks, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    Decomposition decomposition(Grid(original), Grid(enlarged), rank, numtasks);
    ASSERT_EQ(decomposition.getMainFlow(), main);
    EXPECT_EQ(decomposition.getWaveletFlow(), wavelet);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
    return RUN_ALL_TESTS();
}