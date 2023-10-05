#pragma once
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>

class MpiHelper
{
public:
	std::vector<double> MainFlowForRank(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& extension, int rank, int numranks);
	Eigen::VectorXd MpiWaveletFlow(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& wavelet, int rank, int numranks);
	Eigen::VectorXd MpiMainFlow(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& extension, int rank, int numtasks);
	std::vector<double> WaveletFlowForRank(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& wavelet, int rank, int numtasks);
	MpiHelper();
};

