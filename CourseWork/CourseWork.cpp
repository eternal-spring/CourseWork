#include <iostream>
#include "Decomposition.h"
#include "GridGenerator.h"
#include "MpiHelper.h"
#include <functional>
#include <mpi.h>

using namespace std::placeholders;

int main(int argc, char **argv)
{
    int numtasks, rank;
    double begin, end;
    std::vector<double> local_main;
    std::vector<double> total_main;
    std::pair<Grid, Grid> grids;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    std::vector<std::pair<int, int>> sizes = { {500, 50}, {500, 250}, {1000, 250},
        {1000, 500}, {5000, 500},{5000, 2500}, {10000, 1000}, {10000, 5000} };
    {
        for (auto& pair : sizes)
        {
            std::vector<double> original(pair.first);
            std::vector<double> enlarged(pair.second);
            int original_size = 0;
            int enlarged_size = 0;
            if (rank == 0)
            {
                std::cout << "Original Grid Size: " << pair.first << " \n";
                std::cout << "Enlarged Grid Size: " << pair.second << " \n";
                grids = GridGenerator::SeedTestGrids(pair.first, pair.second);
                auto grids_first = grids.first.getValues();
                original = std::vector<double>(&grids_first[0], grids_first.data() + grids_first.cols() * grids_first.rows());
                auto grids_second = grids.second.getValues();
                enlarged = std::vector<double>(&grids_second[0], grids_second.data() + grids_second.cols() * grids_second.rows());
                original_size = original.size();
                enlarged_size = enlarged.size();
            }
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(&original_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&enlarged_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            original.resize(original_size);
            enlarged.resize(enlarged_size);
            MPI_Bcast(original.data(), original_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(enlarged.data(), enlarged_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank != 0) {
                Grid first(original);
                Grid second(enlarged);
                grids = { first, second };
            }
            begin = MPI_Wtime();
            Decomposition decomposition(grids.first, grids.second, rank, numtasks);
            end = MPI_Wtime();
            if (rank == 0) {
                std::cout << "Decomposition time: " << end - begin << " s\n";
            }
        }
    }

    MPI_Finalize();

    return 0;
}
