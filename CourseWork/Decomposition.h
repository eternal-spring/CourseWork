#include<vector>
#include "Grid.h"
#include<Eigen/Dense>
#include<Eigen/Sparse>

using namespace Eigen;

class Decomposition
{
private:
	VectorXd mOriginalFlow;
	VectorXd mMainFlow;
	VectorXd mWaveletFlow;
	SparseMatrix<int> mEmbeddingMatrix;
	SparseMatrix<int> mExtensionMatrix;
	void setOriginalFlow(Grid& original);	
	void FindEmbeddingAndExtensionMatrix(Grid& original, Grid& enlarged);
	void FindMainFlow(bool parallel);
	void FindWaveletFlow(bool parallel);


public:
	Decomposition(Grid originalGrid, Grid enlargedGrid, bool parallel);

	VectorXd getOriginalFlow() { return mOriginalFlow; }
	VectorXd getMainFlow() { return mMainFlow; }
	VectorXd getWaveletFlow() { return mWaveletFlow; }
	SparseMatrix<int> getEmbeddingMatrix() { return mEmbeddingMatrix; }
	SparseMatrix<int> getExtensionMatrix() { return mExtensionMatrix; }
};

