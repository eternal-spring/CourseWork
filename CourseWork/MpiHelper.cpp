#include "MpiHelper.h"

std::vector<double> MpiHelper::EigenToStd(Eigen::VectorXd EigenVector) {
	return std::vector<double>(&EigenVector[0], EigenVector.data() + EigenVector.cols() * EigenVector.rows());
}

Eigen::VectorXd MpiHelper::MpiMainFlow(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& extension, int rank, int numtasks) {
	const std::vector<double> local_main = MainFlowForRank(original, extension, rank, numtasks);
	int size = extension.rows();
	std::vector<double> total_main(size);
	MPI_Reduce(local_main.data(), total_main.data(),size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(total_main.data(), total_main.size());
}

std::vector<double> MpiHelper::WaveletFlowForRank(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& wavelet, int rank, int ranks_num)
{
	auto& wavelet_matrix = wavelet;
	auto& original_flow = original;
	Eigen::VectorXd local_wavelet = Eigen::VectorXd::Zero(wavelet_matrix.rows());
	auto partition = wavelet_matrix.rows() / ranks_num;
	for (int i = partition * rank; i < partition * rank + partition; ++i) {
		for (int j = 0; j < wavelet_matrix.cols(); ++j) {
			local_wavelet(i) += original_flow(j) * wavelet_matrix.coeff(i, j);
		}
	}
	return EigenToStd(local_wavelet);
}

MpiHelper::MpiHelper()
{
}

std::vector<double> MpiHelper::MainFlowForRank(Eigen::VectorXd original, Eigen::SparseMatrix<int> extension, int rank, int ranks_num) {
	auto& extension_matrix = extension;
	auto& original_flow = original;
	Eigen::VectorXd local_main(extension_matrix.rows());
	auto partition = extension_matrix.rows() / ranks_num;
	for (int i = partition * rank; i < partition * rank + partition; ++i) {
		for (int j = 0; j < extension_matrix.cols(); ++j) {
			if (extension_matrix.coeff(i, j) == 1) {
				local_main(i) = original_flow(j);
				break;
			}
		}
	}
	return EigenToStd(local_main);
}

Eigen::VectorXd MpiHelper::MpiWaveletFlow(Eigen::VectorXd original, Eigen::SparseMatrix<int> wavelet, int rank, int numranks)
{
	const std::vector<double> local_main = WaveletFlowForRank(original, wavelet, rank, numranks);
	int size = wavelet.rows();
	std::vector<double> total_main(size);
	MPI_Reduce(local_main.data(), total_main.data(), size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(total_main.data(), total_main.size());
}

