#include "MpiHelper.h"

Eigen::VectorXd MpiHelper::MpiMainFlow(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& extension, int rank, int numtasks) {
	int size = extension.rows();
	std::vector<double> local_main = MainFlowForRank(original, extension, rank, numtasks);
	std::vector<double> total_main(size);
	MPI_Reduce(&local_main[0], &total_main[0], size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(&total_main[0], size);
}

std::vector<double> MpiHelper::WaveletFlowForRank(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& wavelet, int rank, int ranks_num)
{
	auto& wavelet_matrix = wavelet;
	auto& original_flow = original;
	int size = wavelet_matrix.rows();
	std::vector<double> local_wavelet(size, 0.0);
	auto partition = size / ranks_num;
	for (int i = partition * rank; i < partition * rank + partition; ++i) {
		if (i == size) break;
		for (int j = 0; j < wavelet_matrix.cols(); ++j) {
			local_wavelet.at(i) += original_flow(j) * wavelet_matrix.coeff(i, j);
		}
	}
	return local_wavelet;
}

MpiHelper::MpiHelper()
{
}

std::vector<double> MpiHelper::MainFlowForRank(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& extension, int rank, int ranks_num) {
	auto& extension_matrix = extension;
	auto& original_flow = original;
	int size = extension_matrix.rows();
	std::vector<double> local_main(size, 0.0);
	auto partition = size / ranks_num;
	for (int i = partition * rank; i < partition * rank + partition; ++i) {
		if (i == size) break;
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
	std::vector<double> local_main = WaveletFlowForRank(original, wavelet, rank, numranks);
	std::vector<double> total_main(size);
	MPI_Reduce(&local_main[0], &total_main[0], size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(total_main.data(), total_main.size());
}
