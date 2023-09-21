#include <iostream>
#include "Decomposition.h"
#include "GridGenerator.h"
#include <chrono>
#include <functional>
using namespace std::placeholders;

int main()
{
    std::vector<std::pair<int, int>> sizes = { {500, 50}, {500, 250}, {1000, 250},
                                                {1000, 500}, {5000, 1000}, {5000, 2500} };
    for (auto &pair : sizes) {
        std::cout << "Original Grid Size: " << pair.first << " \n";
        std::cout << "Enlarged Grid Size: " << pair.second << " \n";
        auto grids = GridGenerator::SeedTestGrids(pair.first, pair.second);
        auto begin = std::chrono::steady_clock::now();
        Decomposition decomposition(grids.first, grids.second, false);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        std::cout << "Non-parallel Decomposition Time: " << elapsed_ms.count() << " ms\n";
        begin = std::chrono::steady_clock::now();
        Decomposition parallel_decomposition(grids.first, grids.second, true);
        end = std::chrono::steady_clock::now();
        elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        std::cout << "Parallel Decomposition Time: " << elapsed_ms.count() << " ms\n";
    }
}
