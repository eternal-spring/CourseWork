#include<vector>
#include "Grid.h"
#include<Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;
using matrix = MatrixXi;
using flow = VectorXd;

class Decomposition
{
private:
	flow mOriginalFlow;
	flow mMainFlow;
	flow mWaveletFlow;
	matrix mEmbeddingMatrix;
	matrix mExtensionMatrix;
	void setOriginalFlow(Grid& original);	
	void FindEmbeddingAndExtensionMatrix(Grid& original, Grid& enlarged);
	void FindMainFlow(bool parallel);
	void FindWaveletFlow(bool parallel);


public:
	Decomposition(Grid originalGrid, Grid enlargedGrid, bool parallel);

	flow getOriginalFlow() { return mOriginalFlow; }
	flow getMainFlow() { return mMainFlow; }
	flow getWaveletFlow() { return mWaveletFlow; }
	matrix getEmbeddingMatrix() { return mEmbeddingMatrix; }
	matrix getExtensionMatrix() { return mExtensionMatrix; }
};

