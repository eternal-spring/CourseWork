#include "Grid.h"
#include <random>
#pragma once
class GridGenerator
{
public:
	static std::pair<Grid, Grid> SeedTestGrids(int originalSize, int enlargedSize);
};

