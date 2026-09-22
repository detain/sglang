// Fused Qwen4-Exp hyper-connection combine + next-block grouped Gemma RMSNorm.
//
// Byte-parity contract (S16): this kernel must produce, bit-for-bit, the same
// `updated` and `normalized` tensors as running
//   elementwise/hc_combine.cuh (hc_combine_kernel)   then
//   elementwise/grouped_gemma_rmsnorm.cuh (grouped_gemma_rmsnorm_kernel)
// with group size == kHiddenSize (the `hc_per_branch_norm=True` layout).
// Every arithmetic expression below is replicated verbatim from those two
// kernels so the same nvcc (same DEFAULT_CFLAGS: -std=c++20 -O3, no fast-math
// on sm>=100, jit/utils/arch.py get_activation_cuda_cflags is opt-in only)
// emits the same contraction choices and the same reduction trees.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

struct HcCombineNormQwenParams {
  const void* block_output;     // [M, H]
  const void* residual;         // [M, HC * H]
  const void* normed_residual;  // [M, HC * H]
  const void* inject_weight;    // [HC, HC * H]
  const void* norm_weight;      // [HC * H]  (the CONSUMING block's hc_norm weight)
  void* updated;                // [M, HC * H] raw_new
  void* normalized;             // [M, HC * H] normed_new
  float eps;
};

/**
 * \brief Combine the gated residual stream and normalize it in one pass.
 *
 *   a[m, c]      = 2 * sigmoid(dot(normed[m, :], inject_weight[c, :]) / kHcCount)
 *   updated[m, c*H + i] = bf16(residual + a * block_output)                      (phase 1-2)
 *   normalized = grouped RMSNorm(updated) with norm_weight, group = kHiddenSize  (phase 3)
 *
 * The norm reads the *rounded* bf16 `updated` values, exactly as the unfused
 * path does when it re-reads `updated` from global memory; here the rounded
 * values are staged through shared memory instead (`collapsed`), which keeps
 * bit-parity while removing the 84 MB/4096-row write-then-read round trip.
 *
 * Layout notes for bit-parity:
 * - Phase 1/2 replicate `hc_combine_kernel` with the identical 256-thread,
 *   5x16B-per-thread mapping, per-thread fp32 accumulation order, warp
 *   butterfly and 8-warp sequential smem sum, so `a` is bit-identical.
 * - Phase 3 replicates `grouped_gemma_rmsnorm_kernel` (one branch == one
 *   variance group). Unfused it runs 160 threads x {32B, 1 load} on Blackwell
 *   and 160 x {16B, 2 loads} before; here the same tree is replayed by the
 *   first 160 threads of the 256-thread CTA once per owned branch, reading the
 *   staged row from smem. The element -> lane -> tree-node mapping is identical
 *   to the unfused kernel, so the fp32 variance sum is bit-identical.
 * - grid.y splits the kHcCount branches across `parts` CTAs per row for small
 *   batches (redundant gate dot, disjoint outputs); every part computes the
 *   same `a` bit-for-bit.
 *
 * \tparam kHcCount    Number of hyper-connection branches (4 in production).
 * \tparam kHiddenSize Per-branch hidden size H == norm group size; must make
 *                     kRowSize map exactly onto 256 x 5 16B vectors and be a
 *                     multiple of 512 with H/16 <= 256.
 * \tparam kUsePDL     Whether to emit the PDL wait/trigger pair.
 * \tparam Float       Element type: bf16_t | fp16_t.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(256) void hc_combine_norm_qwen_kernel(
    const HcCombineNormQwenParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;  // 8 elements, 16 bytes
  constexpr uint32_t kVecLen = 8;
  constexpr uint32_t kNumThreads = 256;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;            // 1280 for 4x2560
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kNumThreads;  // 5 for 4x2560
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;      // 320 for 2560
  constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;

  // Phase-3 (grouped_gemma_rmsnorm.cuh) layout, replicated per branch.
  constexpr int64_t kGroupSize = kHiddenSize;
#if SGL_ARCH_BLACKWELL_OR_GREATER
  using NStorage = AlignedVector<Float2, 8>;  // 32B smem ops are not portable; staged 2x16B below
  constexpr uint32_t kNumNLoads = 1;
  constexpr uint32_t kNVecLen = 8;
#else
  using NStorage = AlignedVector<Float2, 4>;
  constexpr uint32_t kNumNLoads = 2;
  constexpr uint32_t kNVecLen = 4;
#endif
  constexpr uint32_t kNumNThreads = kGroupSize / 16;  // 160 for 2560
  constexpr uint32_t kNumNWarps = kNumNThreads / kWarpThreads;
  static_assert(kNumNThreads % kWarpThreads == 0, "norm group must divide into full warps");
  static_assert(kNumNThreads <= kNumThreads, "fused CTA must host the whole norm tree");

  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  const uint32_t m = blockIdx.x;
  const uint32_t parts = gridDim.y;
  const uint32_t branchesPerPart = kHcCount / parts;
  const uint32_t branchBase = blockIdx.y * branchesPerPart;

  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto n_ptr = pointer::offset<Float>(params.normed_residual, static_cast<int64_t>(m) * kRowSize);
  const auto w_ptr = static_cast<const Float*>(params.inject_weight);
  const auto updated_ptr = pointer::offset<Float>(params.updated, static_cast<int64_t>(m) * kRowSize);
  const auto normalized_ptr = pointer::offset<Float>(params.normalized, static_cast<int64_t>(m) * kRowSize);

  // Rounded `updated` row staged for the norm phase (20 KB for 4x2560 bf16).
  __shared__ __align__(16) Float collapsed[kRowSize];

  PDLWaitPrimary<kUsePDL>();

  // ---- Phase 1: gate values, verbatim hc_combine.cuh:64-108 ----
  Storage n_vec[kVecsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    n_vec[j] = gmem.load(n_ptr, j);
  }

  float acc[kHcCount];
#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    const auto wc_ptr = w_ptr + static_cast<int64_t>(c) * kRowSize;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const Storage w_vec = gmem.load(wc_ptr, j);
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
        sum += nx * wx + ny * wy;
      }
    }
    acc[c] = warp::reduce_sum(sum);
  }

  __shared__ float smem[kHcCount][kNumWarps];
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < kHcCount; ++c) {
      smem[c][warp_id] = acc[c];
    }
  }
  __syncthreads();
  __shared__ float a_shared[kHcCount];
  if (threadIdx.x < kHcCount) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) {
      total += smem[threadIdx.x][w];
    }
    a_shared[threadIdx.x] = 2.0f / (1.0f + math::exp(-total / kHcCount));
  }
  __syncthreads();

  // ---- Phase 2: combine owned branches, verbatim hc_combine.cuh:112-130 ----
  // plus a 16B smem store of the rounded value for phase 3.
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
    const uint32_t branch = vec_idx / kVecsPerBranch;
    if (branch < branchBase || branch >= branchBase + branchesPerPart) {
      continue;
    }
    const uint32_t col_in_branch = (vec_idx % kVecsPerBranch) * kVecLen;
    const float a = a_shared[branch];

    const Storage r_vec = gmem.load(r_ptr, j);
    Storage y_vec;
    y_vec.load(y_ptr, col_in_branch / kVecLen);
    Storage out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);
      out_vec[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    gmem.store(updated_ptr, out_vec, j);
    out_vec.store(collapsed, vec_idx);  // 16B smem store at the same linear slot
  }
  __syncthreads();

  // ---- Phase 3: grouped RMSNorm per owned branch, verbatim
  // grouped_gemma_rmsnorm.cuh:56-116 replayed by threads [0, kNumNThreads) ----
  constexpr uint32_t kF2PerBranch = kGroupSize / 2;  // Float2 units per branch
  __shared__ float nsmem[kWarpThreads];
  const auto nw_gmem = tile::Memory<NStorage>::cta(kNumNThreads);
  const auto weight_base = static_cast<const Float*>(params.norm_weight);
  for (uint32_t b = 0; b < branchesPerPart; ++b) {
    const uint32_t group = branchBase + b;
    // input/weight tiles stay live across the cross-warp reduce (the 160
    // active threads hold them; the idle 96 never read them).
    NStorage input_vec[kNumNLoads];
    NStorage weight_vec[kNumNLoads];
    float sum_of_squares = 0.0f;
    if (threadIdx.x < kNumNThreads) {
      const auto w_ptr_g = pointer::offset<Float>(weight_base, static_cast<int64_t>(group) * kGroupSize);
#pragma unroll
      for (uint32_t j = 0; j < kNumNLoads; ++j) {
        // smem staging caps at 16B: fill the (possibly 32B) input vector with
        // two half-vector copies; same values, only the load granularity moves.
        const int64_t f2_base =
            static_cast<int64_t>(group) * kF2PerBranch + (threadIdx.x + j * kNumNThreads) * kNVecLen;
        const auto* sm_f2 = reinterpret_cast<const Float2*>(collapsed);
#pragma unroll
        for (uint32_t k = 0; k < kNVecLen / 4; ++k) {
          // 16B chunks: kNVecLen/4 sub-vectors of 4 Float2 each
          AlignedVector<Float2, 4> part;
          part.load(sm_f2 + f2_base + k * 4);
#pragma unroll
          for (uint32_t e = 0; e < 4; ++e) {
            input_vec[j][k * 4 + e] = part[e];
          }
        }
        weight_vec[j] = nw_gmem.load(w_ptr_g, j);
      }

#pragma unroll
      for (uint32_t j = 0; j < kNumNLoads; ++j) {
#pragma unroll
        for (uint32_t i = 0; i < kNVecLen; ++i) {
          const auto [x, y] = cast<fp32x2_t>(input_vec[j][i]);
          sum_of_squares += x * x + y * y;
        }
      }
      sum_of_squares = warp::reduce_sum(sum_of_squares);
    }
    float norm_factor;
    if constexpr (kNumNWarps == 1) {
      if (threadIdx.x < kNumNThreads) {
        norm_factor = math::rsqrt(sum_of_squares / kGroupSize + params.eps);
      }
    } else {
      __syncthreads();  // protect nsmem reads from the previous branch
      if (threadIdx.x < kNumNThreads) {
        nsmem[warp_id] = sum_of_squares;
      }
      __syncthreads();
      if (warp_id == 0) {
        const auto tx = threadIdx.x;
        const auto local_sum = tx < kNumNWarps ? nsmem[tx] : 0.0f;
        const float total = warp::reduce_sum(local_sum);
        nsmem[tx] = math::rsqrt(total / kGroupSize + params.eps);
      }
      __syncthreads();
      if (threadIdx.x < kNumNThreads) {
        norm_factor = nsmem[warp_id];
      }
    }
    if (threadIdx.x < kNumNThreads) {
      const auto out_ptr_g =
          pointer::offset<Float>(params.normalized, static_cast<int64_t>(m) * kRowSize + static_cast<int64_t>(group) * kGroupSize);
#pragma unroll
      for (uint32_t j = 0; j < kNumNLoads; ++j) {
        NStorage output_vec;
#pragma unroll
        for (uint32_t i = 0; i < kNVecLen; ++i) {
          const auto [ix, iy] = cast<fp32x2_t>(input_vec[j][i]);
          const auto [wx, wy] = cast<fp32x2_t>(weight_vec[j][i]);
          output_vec[i] = cast<Float2>(fp32x2_t{ix * norm_factor * (1.0f + wx), iy * norm_factor * (1.0f + wy)});
        }
        nw_gmem.store(out_ptr_g, output_vec, j);
      }
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcCombineNormQwenKernel {
  static_assert(sizeof(DType) == 2, "HcCombineNormQwen only supports 2-byte dtypes");
  static_assert(kHcCount > 0, "kHcCount must be positive");
  static_assert(kHcCount % 4 == 0, "parts heuristic requires kHcCount divisible by 4");
  static_assert(kHiddenSize > 0 && kHiddenSize % 8 == 0, "kHiddenSize must be a multiple of 8");
  static_assert((kHcCount * kHiddenSize) % (256 * 8) == 0, "kHcCount * kHiddenSize must be a multiple of 2048");
  static_assert(kHiddenSize % 512 == 0, "kHiddenSize (norm group) must be a multiple of 512");
  static_assert(kHiddenSize / 16 <= 256, "norm group tree must fit the 256-thread CTA");
  static constexpr auto kernel = hc_combine_norm_qwen_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = 256;

  /**
   * \brief Validate tensors and launch the fused combine+norm.
   * \param block_output    [M, H] contiguous
   * \param residual        [M, HC * H] contiguous
   * \param normed_residual [M, HC * H] contiguous
   * \param inject_weight   [HC, HC * H] contiguous
   * \param norm_weight     [HC * H] contiguous (consuming block's hc_norm weight)
   * \param updated         [M, HC * H] out, raw_new
   * \param normalized      [M, HC * H] out, normed_new
   * \param eps             RMSNorm epsilon
   */
  static void run(const tvm::ffi::TensorView block_output,
                  const tvm::ffi::TensorView residual,
                  const tvm::ffi::TensorView normed_residual,
                  const tvm::ffi::TensorView inject_weight,
                  const tvm::ffi::TensorView norm_weight,
                  tvm::ffi::TensorView updated,
                  tvm::ffi::TensorView normalized,
                  float eps) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize})  // block_output
        .with_dtype<DType>()
        .with_device(device)
        .verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})  // residual, normed_residual, updated, normalized
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(normed_residual)
        .verify(updated)
        .verify(normalized);
    TensorMatcher({kHcCount, kHcCount * kHiddenSize})  // inject_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(inject_weight);
    TensorMatcher({kHcCount * kHiddenSize})  // norm_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(norm_weight);

    CHECK_HOST(updated.data_ptr() != residual.data_ptr() && updated.data_ptr() != normed_residual.data_ptr() &&
               updated.data_ptr() != block_output.data_ptr())
        << "updated must not alias inputs";
    CHECK_HOST(normalized.data_ptr() != residual.data_ptr() && normalized.data_ptr() != normed_residual.data_ptr() &&
               normalized.data_ptr() != block_output.data_ptr() && normalized.data_ptr() != updated.data_ptr())
        << "normalized must not alias inputs or updated";

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    if (num_tokens == 0) {
      return;  // DP-idle guard; nothing to launch
    }
    // Branch-split for tiny grids: redundant gate dot, disjoint branch outputs.
    const uint32_t parts = num_tokens <= 8 ? 4 : (num_tokens <= 48 ? 2 : 1);

    const auto params = HcCombineNormQwenParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .normed_residual = normed_residual.data_ptr(),
        .inject_weight = inject_weight.data_ptr(),
        .norm_weight = norm_weight.data_ptr(),
        .updated = updated.data_ptr(),
        .normalized = normalized.data_ptr(),
        .eps = eps,
    };

    LaunchKernel(dim3(num_tokens, parts, 1), kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
