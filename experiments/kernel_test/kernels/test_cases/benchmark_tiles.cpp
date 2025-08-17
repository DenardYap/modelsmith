#include <iostream>
#include <torch/torch.h>
#include <chrono>
#include <vector>
#include <map>
#include <tuple>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdint> // Required for uint8_t
#include <algorithm> // Required for std::sort
#include <limits> // Required for std::numeric_limits

// Forward declarations
torch::Tensor pack_binary_matrix_SIMD128(torch::Tensor mat);
void neon_binary_gemm_tiled_with_params(const uint8_t* A_packed,
                                       const uint8_t* B_packed,
                                       int32_t* C,
                                       int M, int K, int N,
                                       int tile_m, int tile_n, int tile_k);

// Structure to hold tile configuration and its timing
struct TileConfig {
    int tile_m;
    int tile_n;
    int tile_k;
    double time_ms;
    
    // Default constructor required for std::map
    TileConfig() : tile_m(16), tile_n(128), tile_k(128), time_ms(std::numeric_limits<double>::max()) {}
    
    TileConfig(int m, int n, int k, double t) 
        : tile_m(m), tile_n(n), tile_k(k), time_ms(t) {}
    
    // For sorting configs by time
    bool operator<(const TileConfig& other) const {
        return time_ms < other.time_ms;
    }
};

// Structure for matrix dimensions as key
struct MatrixDims {
    int M, N, K;
    
    MatrixDims() : M(0), N(0), K(0) {}
    MatrixDims(int m, int n, int k) : M(m), N(n), K(k) {}
    
    bool operator<(const MatrixDims& other) const {
        return std::tie(M, N, K) < std::tie(other.M, other.N, other.K);
    }
};

// Measure performance for a specific tile configuration
double benchmark_tile_config(int M, int K, int N, int tile_m, int tile_n, int tile_k, int trials = 5) {
    torch::manual_seed(42);  // For reproducibility
    
    // Generate random binary matrices (-1, 1)
    auto A = (torch::randint(0, 2, {M, K}, torch::kFloat32) * 2 - 1).to(torch::kFloat32);
    auto B = (torch::randint(0, 2, {K, N}, torch::kFloat32) * 2 - 1).to(torch::kFloat32);

    // Convert to int8 and pack
    auto A_int8 = A.to(torch::kInt8).contiguous();
    auto B_int8 = B.to(torch::kInt8).contiguous();
    auto B_t_int8 = B_int8.transpose(0, 1).contiguous();
    
    auto A_packed = pack_binary_matrix_SIMD128(A_int8);
    auto B_packed = pack_binary_matrix_SIMD128(B_t_int8);
    
    // Warmup run
    {
        auto C_out = torch::zeros({M, N}, torch::kInt32);
        neon_binary_gemm_tiled_with_params(
            A_packed.data_ptr<uint8_t>(),
            B_packed.data_ptr<uint8_t>(),
            C_out.data_ptr<int32_t>(),
            M, K, N, tile_m, tile_n, tile_k
        );
    }
    
    // Timed runs
    std::vector<double> times;
    times.reserve(trials);
    
    for (int i = 0; i < trials; i++) {
        auto C_out = torch::zeros({M, N}, torch::kInt32);
        auto start = std::chrono::high_resolution_clock::now();
        
        neon_binary_gemm_tiled_with_params(
            A_packed.data_ptr<uint8_t>(),
            B_packed.data_ptr<uint8_t>(),
            C_out.data_ptr<int32_t>(),
            M, K, N, tile_m, tile_n, tile_k
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    
    // Return median time (more robust than mean)
    std::sort(times.begin(), times.end());
    return times[trials/2];
}

// Convert tile config to JSON string
std::string tile_config_to_json(const MatrixDims& dims, const TileConfig& config) {
    std::stringstream ss;
    ss << "  \"" << dims.M << "," << dims.N << "," << dims.K << "\": {\n"
       << "    \"tile_m\": " << config.tile_m << ",\n"
       << "    \"tile_n\": " << config.tile_n << ",\n"
       << "    \"tile_k\": " << config.tile_k << ",\n"
       << "    \"time_ms\": " << std::fixed << std::setprecision(4) << config.time_ms << "\n"
       << "  }";
    return ss.str();
}

int main() {
    // Matrix dimensions to test
    std::vector<int> dims = {1, 16, 64, 128, 256, 512, 1024, 2048, 4096};
    
    // Tile sizes to test (powers of 2, favoring cache-friendly sizes)
    std::vector<int> tile_dims = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    
    // Map to store best config for each matrix size
    std::map<MatrixDims, TileConfig> best_configs;
    
    // Progress tracking
    int total_combos = dims.size() * dims.size() * dims.size();
    int current_combo = 0;
    
    // Test all matrix dimension combinations
    for (int M : dims) {
        for (int K : dims) {
            for (int N : dims) {
                current_combo++;
                std::cout << "\rTesting matrix " << current_combo << "/" << total_combos 
                          << ": " << M << "x" << K << " * " << K << "x" << N << std::flush;
                
                std::vector<TileConfig> configs;
                configs.reserve(tile_dims.size() * tile_dims.size() * tile_dims.size());
                
                // Test different tile configurations
                for (int tm : tile_dims) {
                    if (tm > M) continue;  // Skip if tile size > matrix dim
                    
                    for (int tn : tile_dims) {
                        if (tn > N) continue;
                        
                        for (int tk : tile_dims) {
                            if (tk > K) continue;
                            
                            // Ensure tk is multiple of 8 for binary packing
                            int aligned_tk = ((tk + 7) / 8) * 8;
                            
                            try {
                                double time = benchmark_tile_config(M, K, N, tm, tn, aligned_tk);
                                configs.emplace_back(tm, tn, aligned_tk, time);
                            } catch (const std::exception& e) {
                                std::cerr << "\nError with tiles (" << tm << "," << tn << "," 
                                         << aligned_tk << "): " << e.what() << std::endl;
                                continue;
                            }
                        }
                    }
                }
                
                // Find best config
                if (!configs.empty()) {
                    auto best = std::min_element(configs.begin(), configs.end());
                    best_configs[MatrixDims(M, N, K)] = *best;
                    
                    // Print if significantly better than default
                    TileConfig default_config;
                    if (best->time_ms < default_config.time_ms * 0.8) {  // 20% better
                        std::cout << "\nFound good config for " << M << "x" << K << " * " << K << "x" << N 
                                  << " -> tiles(" << best->tile_m << "," << best->tile_n << "," 
                                  << best->tile_k << ") = " << best->time_ms << "ms" << std::endl;
                    }
                }
            }
        }
    }
    std::cout << "\nDone testing configurations." << std::endl;
    
    // Write results to JSON file
    std::ofstream json_file("optimal_tiles.json");
    json_file << "{\n";
    
    bool first = true;
    for (const auto& [dims, config] : best_configs) {
        if (!first) json_file << ",\n";
        json_file << tile_config_to_json(dims, config);
        first = false;
    }
    
    json_file << "\n}\n";
    json_file.close();
    
    std::cout << "Wrote optimal tile configurations to optimal_tiles.json" << std::endl;
    
    // Print some example best configurations
    std::cout << "\nExample optimal configurations:" << std::endl;
    int examples = 0;
    for (const auto& [dims, config] : best_configs) {
        if (examples++ >= 5) break;
        std::cout << dims.M << "x" << dims.K << " * " << dims.K << "x" << dims.N 
                  << " -> tiles(" << config.tile_m << "," << config.tile_n << "," 
                  << config.tile_k << ") = " << config.time_ms << "ms" << std::endl;
    }
    
    return 0;
} 