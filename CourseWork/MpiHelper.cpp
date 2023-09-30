#include "MpiHelper.h"

std::vector<double> MpiHelper::EigenToStd(Eigen::VectorXd EigenVector) {
	return std::vector<double>(&EigenVector[0], EigenVector.data() + EigenVector.cols() * EigenVector.rows());
}

Eigen::VectorXd MpiHelper::MpiMainFlow(Eigen::VectorXd& original, Eigen::SparseMatrix<int>& extension, int rank, int numtasks) {
	//int numtasks, rank;
	const std::vector<double> local_main = MainFlowForRank(original, extension, rank, numtasks);
	std::vector<double> *total_main = new std::vector<double>(local_main.size());
	//std::vector<double>& total_main_address = total_main;
	int size = extension.rows();
	//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	//local_main = MainFlowForRank(original, extension, rank, numtasks);
	MPI_Reduce(&local_main, &(*total_main), local_main.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	//MPI_Reduce_local(&local_main, &total_main, size, MPI_DOUBLE, MPI_SUM);
	//std::vector<double> main(*total_main);
	//auto main = total_main;
	//return main;

	Eigen::VectorXd main;
	main = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(total_main->data(), total_main->size());
	return main;
}

MpiHelper::MpiHelper()
{
}

//MpiHelper::~MpiHelper()
//{
//}


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

