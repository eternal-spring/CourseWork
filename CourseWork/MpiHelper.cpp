#include "MpiHelper.h"

MpiHelper::MpiHelper()
{
}

Eigen::VectorXd MpiHelper::MpiMainFlow(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& extension, int rank, int numtasks) {
	int size = extension.rows();
	std::vector<double> local_main = MainFlowForRank(original, extension, rank, numtasks);
	std::vector<double> total_main(size);
	MPI_Reduce(local_main.data(), total_main.data(), size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(total_main.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(total_main.data(), size);
}

std::vector<double> MpiHelper::WaveletFlowForRank(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& wavelet, int rank, int ranks_num)
{
	auto& wavelet_matrix = wavelet;
	auto& original_flow = original;
	int size = wavelet_matrix.rows();
	std::vector<double> local_wavelet(size, 0.0);
	auto partition = size / ranks_num;
	int lower_bound = partition * rank;
	int upper_bound = rank == ranks_num - 1 ? size : lower_bound + partition;
	for (int i = lower_bound; i < upper_bound; ++i) {
		for (int j = 0; j < wavelet_matrix.cols(); ++j) {
			local_wavelet.at(i) += original_flow(j) * wavelet_matrix.coeff(i, j);
		}
	}
	return local_wavelet;
}


std::vector<double> MpiHelper::MainFlowForRank(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& extension, int rank, int ranks_num) {
	auto& extension_matrix = extension;
	auto& original_flow = original;
	int size = extension_matrix.rows();
	std::vector<double> local_main(size, 0.0);
	auto partition = size / ranks_num;
	int lower_bound = partition * rank;
	int upper_bound = rank == ranks_num - 1 ? size : lower_bound + partition;
	for (int i = lower_bound; i < upper_bound; ++i) {
		for (int j = 0; j < extension_matrix.cols(); ++j) {
			if (extension_matrix.coeff(i, j) == 1) {
				local_main.at(i) = original_flow(j);
				break;
			}
		}
	}
	return local_main;
}

Eigen::VectorXd MpiHelper::MpiWaveletFlow(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& wavelet, int rank, int numranks)
{
	int size = wavelet.rows();
	std::vector<double> local_wavelet = WaveletFlowForRank(original, wavelet, rank, numranks);
	std::vector<double> total_wavelet(size);
	MPI_Reduce(&local_wavelet[0], &total_wavelet[0], size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(total_wavelet.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(total_wavelet.data(), total_wavelet.size());
}
