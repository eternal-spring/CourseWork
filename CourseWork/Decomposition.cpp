#include "Decomposition.h"
#include "MatrixCalculator.h"

Decomposition::Decomposition(Grid originalGrid, Grid enlargedGrid, bool parallel) {
	setOriginalFlow(originalGrid);
	FindEmbeddingAndExtensionMatrix(originalGrid, enlargedGrid);
	FindMainFlow(parallel);
	FindWaveletFlow(parallel);
}

void Decomposition::setOriginalFlow(Grid& original) {
	auto flow = original.getValues();
	mOriginalFlow.assign(flow.begin(), flow.end() - 1);
}

void Decomposition::FindEmbeddingAndExtensionMatrix(Grid& original, Grid& enlarged) {
	auto enlarged_flow = enlarged.getValues();
	auto original_flow = original.getValues();
	int columns_count = original_flow.size() - 1;
	int rows_count = enlarged_flow.size() - 1;
	matrix embedding_matrix(rows_count, std::vector<int>(columns_count));
	matrix extension_matrix(rows_count, std::vector<int>(columns_count));
	int row = 0;
	embedding_matrix.at(0).at(0) = 1;
	extension_matrix.at(0).at(0) = 1;
	for (int i = 1; i < columns_count; i++)
	{
		if (std::find(enlarged_flow.begin(), enlarged_flow.end(),
			original_flow.at(i)) != enlarged_flow.end()) {
			row++;
			extension_matrix.at(row).at(i) = 1;
		}
		embedding_matrix.at(row).at(i) = 1;
	}
	mEmbeddingMatrix = embedding_matrix;
	mExtensionMatrix = extension_matrix;
}

void Decomposition::FindMainFlow(bool parallel) {
	auto extension_matrix = getExtensionMatrix();
	auto original_flow = getOriginalFlow();
	flow main(extension_matrix.size());
#pragma omp parallel for if (parallel)
	for (int i = 0; i < extension_matrix.size(); ++i) {
		for (int j = 0; j < extension_matrix.at(0).size(); ++j) {
			if (extension_matrix.at(i).at(j) == 1) {
				main.at(i) = original_flow.at(j);
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
	flow wavelet(extension_matrix.at(0).size());
	matrix wavelet_matrix = MatrixCalculator::Subtract(MatrixCalculator::Identity(extension_matrix.at(0).size()), 
		MatrixCalculator::Multiply(MatrixCalculator::Transpose(embedding_matrix), extension_matrix, parallel));
#pragma omp parallel for if (parallel)
	for (int i = 0; i < wavelet_matrix.size(); ++i) {
		for (int j = 0; j < wavelet_matrix.at(0).size(); ++j) {
			wavelet.at(i) += original_flow.at(j) * wavelet_matrix.at(i).at(j);
		}
	}
	mWaveletFlow = wavelet;
}