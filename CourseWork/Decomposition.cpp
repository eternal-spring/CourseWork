#include "Decomposition.h"

Decomposition::Decomposition(Grid originalGrid, Grid enlargedGrid, bool parallel) {
	setOriginalFlow(originalGrid);
	FindEmbeddingAndExtensionMatrix(originalGrid, enlargedGrid);
	FindMainFlow(parallel);
	FindWaveletFlow(parallel);
}

void Decomposition::setOriginalFlow(Grid& original) {
	auto flow = original.getValues();
	mOriginalFlow = Map<VectorXd, Unaligned>(flow.data(), flow.size());
}

void Decomposition::FindEmbeddingAndExtensionMatrix(Grid& original, Grid& enlarged) {
	auto enlarged_flow = enlarged.getValues();
	auto original_flow = original.getValues();
	const int columns_count = original_flow.rows() - 1;
	const int rows_count = enlarged_flow.rows() - 1;
	SparseMatrix<int> embedding_matrix(rows_count, columns_count);
	SparseMatrix<int> extension_matrix(rows_count, columns_count);
	int row = 0;
	embedding_matrix.insert(0, 0) = 1;
	extension_matrix.insert(0, 0) = 1;
	for (int i = 1; i < columns_count; i++)
	{
		if (enlarged_flow(row + 1) == original_flow(i)) {
			row++;
			extension_matrix.insert(row, i) = 1;
		}
		embedding_matrix.insert(row, i) = 1;
	}
	mEmbeddingMatrix = embedding_matrix;
	mExtensionMatrix = extension_matrix;
}

void Decomposition::FindMainFlow(bool parallel) {
	auto extension_matrix = getExtensionMatrix();
	auto original_flow = getOriginalFlow();
	VectorXd main(extension_matrix.rows());
#pragma omp parallel for if (parallel)
	for (int i = 0; i < extension_matrix.rows(); ++i) {
		for (int j = 0; j < extension_matrix.cols(); ++j) {
			if (extension_matrix.coeff(i, j) == 1) {
				main(i) = original_flow(j);
				break;
			}
		}
	}
	mMainFlow = main;
}

void Decomposition::FindWaveletFlow(bool parallel) {
	auto embedding_matrix = getEmbeddingMatrix();
	auto extension_matrix = getExtensionMatrix();
	auto original_flow = getOriginalFlow();
	VectorXd wavelet = VectorXd::Zero(extension_matrix.cols());
	SparseMatrix<int> identity(extension_matrix.cols(), extension_matrix.cols());
	identity.setIdentity();
	if (parallel) { 
		Eigen::initParallel(); 
		Eigen::setNbThreads(8);
	}
	SparseMatrix<int> wavelet_matrix = identity - embedding_matrix.transpose() * extension_matrix;
#pragma omp parallel for if (parallel)
	for (int i = 0; i < wavelet_matrix.rows(); ++i) {
		for (int j = 0; j < wavelet_matrix.cols(); ++j) {
			wavelet(i) += original_flow(j) * wavelet_matrix.coeff(i,j);
		}
	}
	mWaveletFlow = wavelet;
}