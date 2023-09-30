#include "Decomposition.h"

Decomposition::Decomposition(Grid originalGrid, Grid enlargedGrid, int rank, int numtasks) {
	setOriginalFlow(originalGrid);
	FindEmbeddingAndExtensionMatrix(originalGrid, enlargedGrid);
	FindMainFlow(rank, numtasks);
	FindWaveletFlow(rank, numtasks);
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

void Decomposition::FindMainFlow(int rank, int numtasks) {
	MpiHelper mpi = MpiHelper();
	VectorXd original = getOriginalFlow();
	SparseMatrix<int> extension = getExtensionMatrix();
	auto vector_flow = mpi.MpiMainFlow(original, extension, rank, numtasks);
	mMainFlow = vector_flow;

}

void Decomposition::FindWaveletFlow(int rank, int numtasks) {
	MpiHelper mpi = MpiHelper();
	auto embedding_matrix = getEmbeddingMatrix();
	auto extension_matrix = getExtensionMatrix();
	auto original_flow = getOriginalFlow();
	SparseMatrix<int> identity(extension_matrix.cols(), extension_matrix.cols());
	identity.setIdentity();
	SparseMatrix<int> wavelet_matrix = identity - embedding_matrix.transpose() * extension_matrix;
	auto vector_flow = mpi.MpiWaveletFlow(original_flow, wavelet_matrix, rank, numtasks);
	mWaveletFlow = vector_flow;
}

