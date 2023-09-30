#include<vector>
#include "Grid.h"
#include "MpiHelper.h"
#include<Eigen/Dense>
#include<Eigen/Sparse>

using namespace Eigen;

class Decomposition
{
private:
	VectorXd* mOriginalFlow = new VectorXd;
	VectorXd* mMainFlow = new VectorXd;
	VectorXd* mWaveletFlow = new VectorXd;
	SparseMatrix<int>* mEmbeddingMatrix = new SparseMatrix<int>;
	SparseMatrix<int>* mExtensionMatrix = new SparseMatrix<int>;
	void setOriginalFlow(Grid& original);	
	void FindEmbeddingAndExtensionMatrix(Grid& original, Grid& enlarged);
	void FindMainFlow(int rank, int numtasks);
	void FindWaveletFlow();


public:
	Decomposition(Grid originalGrid, Grid enlargedGrid, int rank, int numtasks);

	VectorXd getOriginalFlow() { return *mOriginalFlow; }
	VectorXd getMainFlow() { return *mMainFlow; }
	VectorXd getWaveletFlow() { return *mWaveletFlow; }
	SparseMatrix<int> getEmbeddingMatrix() { return *mEmbeddingMatrix; }
	SparseMatrix<int> getExtensionMatrix() { return *mExtensionMatrix; }

	Decomposition() = default;
	//~Decomposition();
};

