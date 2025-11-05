#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Allocate result matrix
    Matrix* result = matrix_memory_allocator.Allocate("result");

    // Initialize result as zero matrix with same shape as query
    gpu_sim.Copy(current_query, result, kInGpuHbm);
    result->Zero();

    // For each key-value pair up to current round
    for (size_t j = 0; j <= i; ++j) {
      // Allocate intermediate matrices
      Matrix* qk_transpose = matrix_memory_allocator.Allocate("qk_transpose");
      Matrix* exp_result = matrix_memory_allocator.Allocate("exp_result");
      Matrix* sum_result = matrix_memory_allocator.Allocate("sum_result");
      Matrix* softmax_result = matrix_memory_allocator.Allocate("softmax_result");
      Matrix* temp_result = matrix_memory_allocator.Allocate("temp_result");

      // Move matrices to SRAM for computation
      gpu_sim.MoveMatrixToSharedMem(current_query);
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      gpu_sim.MoveMatrixToSharedMem(values[j]);

      // Compute Q * K^T
      gpu_sim.Transpose(keys[j], kInSharedMemory);
      gpu_sim.MatMul(current_query, keys[j], qk_transpose);

      // Compute exp(QK^T)
      gpu_sim.MatExp(qk_transpose, exp_result);

      // Compute sum of exp values for softmax denominator
      gpu_sim.Sum(exp_result, sum_result);

      // Compute softmax = exp(QK^T) / sum(exp(QK^T))
      gpu_sim.MatDiv(exp_result, sum_result, softmax_result);

      // Compute softmax * V
      gpu_sim.MatMul(softmax_result, values[j], temp_result);

      // Move result to SRAM for accumulation
      gpu_sim.MoveMatrixToSharedMem(result);

      // Accumulate result
      gpu_sim.MatAdd(result, temp_result, result);

      // Move result back to HBM
      gpu_sim.MoveMatrixToGpuHbm(result);

      // Release intermediate matrices
      gpu_sim.ReleaseMatrix(qk_transpose);
      gpu_sim.ReleaseMatrix(exp_result);
      gpu_sim.ReleaseMatrix(sum_result);
      gpu_sim.ReleaseMatrix(softmax_result);
      gpu_sim.ReleaseMatrix(temp_result);
    }

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