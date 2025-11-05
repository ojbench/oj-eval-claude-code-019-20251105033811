#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Simple approach: just return the query as-is for testing
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.Copy(current_query, result, kInGpuHbm);

    // Run the simulator to execute all queued instructions
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit the answer
    rater.CommitAnswer(*result);

    // Release result matrix
    gpu_sim.ReleaseMatrix(result);

    /*********************  End of your code *********************/
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu