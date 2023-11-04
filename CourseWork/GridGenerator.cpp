#include "GridGenerator.h"

std::pair<Grid, Grid> GridGenerator::SeedTestGrids(int originalSize, int enlargedSize) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    std::uniform_real_distribution<double> dist{ 1.0, 1000000.0 };

    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    std::vector<double> original_flow(originalSize);
    generate(begin(original_flow), end(original_flow), gen);
    std::vector<double>::const_iterator first = original_flow.begin();
    std::vector<double>::const_iterator last = original_flow.begin() + enlargedSize - 2;
    std::vector<double> enlarged_flow(first, last);
    auto min_elem = *min_element(original_flow.begin(), original_flow.end());
    auto max_elem = *max_element(original_flow.begin(), original_flow.end());
    enlarged_flow.push_back(min_elem);
    enlarged_flow.push_back(max_elem);
    sort(original_flow.begin(), original_flow.end());
    unique(original_flow.begin(), original_flow.end());
    sort(enlarged_flow.begin(), enlarged_flow.end());
    auto it = unique(enlarged_flow.begin(), enlarged_flow.end());
    enlarged_flow.resize(std::distance(enlarged_flow.begin(), it));
    Grid original(original_flow);
    Grid enlarged(enlarged_flow);
    return { original, enlarged };
}