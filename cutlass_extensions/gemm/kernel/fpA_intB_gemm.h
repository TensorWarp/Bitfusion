#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include <type_traits>


namespace cutlass
{
    namespace gemm
    {
        namespace kernel
        {


            namespace detail
            {
                template <typename>
                inline constexpr bool dependent_false_v = false;
            }

            template <typename Mma_,
                typename Epilogue_,
                typename ThreadblockSwizzle_,
                typename KernelArch,
                bool SplitKSerial
            >
            struct GemmFpAIntB
            {

                using Mma = Mma_;
                using Epilogue = Epilogue_;
                using EpilogueOutputOp = typename Epilogue::OutputOp;
                using ThreadblockSwizzle = ThreadblockSwizzle_;
                static bool const kSplitKSerial = SplitKSerial;

                using ElementA = typename Mma::IteratorA::Element;
                using LayoutA = typename Mma::IteratorA::Layout;
                using ElementB = typename Mma::IteratorB::Element;
                using LayoutB = typename Mma::IteratorB::Element;
                using ElementC = typename Epilogue::OutputTileIterator::Element;
                using LayoutC = typename Mma::LayoutC;
                using ElementScale = ElementC;

                static ComplexTransform const kTransformA = Mma::kTransformA;
                static ComplexTransform const kTransformB = Mma::kTransformA;

                using Operator = typename Mma::Operator;
                using OperatorClass = typename Mma::Operator::OperatorClass;
                using ThreadblockShape = typename Mma::Shape;
                using WarpShape = typename Mma::Operator::Shape;
                using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
                using ArchTag = typename Mma::ArchTag;

                static int const kStages = Mma::kStages;
                static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
                static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
                static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

                using WarpCount = typename Mma::WarpCount;
                static int const kThreadCount = 32 * WarpCount::kCount;

                static constexpr int kInterleave = Mma::IteratorB::Shape::kRow / Mma::Shape::kK;

                struct Arguments
                {
                    GemmUniversalMode mode = GemmUniversalMode::kGemm;

                    cutlass::gemm::GemmCoord problem_size;
                    int group_size;
                    typename Mma::IteratorA::TensorRef ref_A;
                    typename Mma::IteratorB::TensorRef ref_B;
                    typename Mma::IteratorScale::TensorRef ref_scale;
                    typename Mma::IteratorScale::TensorRef ref_zero;
                    typename Epilogue::OutputTileIterator::TensorRef ref_C;
                    typename Epilogue::OutputTileIterator::TensorRef ref_D;

                    int batch_count;

                    typename EpilogueOutputOp::Params output_op;

                    int const* gather_A_indices;
                    int const* gather_B_indices;
                    int const* scatter_D_indices;

                    int batch_stride_D = 0;


                    CUTLASS_HOST_DEVICE
                        Arguments() {}

                    CUTLASS_HOST_DEVICE
                        Arguments(cutlass::gemm::GemmCoord const& problem_size, const int group_size,
                            typename Mma::IteratorA::TensorRef ref_A, typename Mma::IteratorB::TensorRef ref_B,
                            typename Mma::IteratorScale::TensorRef ref_scale, typename Mma::IteratorScale::TensorRef ref_zero,
                            typename Epilogue::OutputTileIterator::TensorRef ref_C,
                            typename Epilogue::OutputTileIterator::TensorRef ref_D, int serial_split_k_factor,
                            typename EpilogueOutputOp::Params output_op = typename EpilogueOutputOp::Params(),
                            int const* gather_A_indices = nullptr, int const* gather_B_indices = nullptr,
                            int const* scatter_D_indices = nullptr)
                        : problem_size(problem_size)
                        , group_size(group_size)
                        , ref_A(ref_A)
                        , ref_B(ref_B)
                        , ref_scale(ref_scale)
                        , ref_zero(ref_zero)
                        , ref_C(ref_C)
                        , ref_D(ref_D)
                        , batch_count(serial_split_k_factor)
                        , output_op(output_op)
                        , gather_A_indices(gather_A_indices)
                        , gather_B_indices(gather_B_indices)
                        , scatter_D_indices(scatter_D_indices)
                    {
                    }
                };

                struct Params
                {
                    cutlass::gemm::GemmCoord problem_size;
                    int group_size;
                    cutlass::gemm::GemmCoord grid_tiled_shape;
                    int swizzle_log_tile;
                    typename Mma::IteratorA::Params params_A;
                    typename Mma::IteratorA::TensorRef ref_A;
                    typename Mma::IteratorB::Params params_B;
                    typename Mma::IteratorB::TensorRef ref_B;
                    typename Mma::IteratorScale::Params params_scale;
                    typename Mma::IteratorScale::TensorRef ref_scale;
                    typename Mma::IteratorScale::TensorRef ref_zero;
                    typename Epilogue::OutputTileIterator::Params params_C;
                    typename Epilogue::OutputTileIterator::TensorRef ref_C;
                    typename Epilogue::OutputTileIterator::Params params_D;
                    typename Epilogue::OutputTileIterator::TensorRef ref_D;
                    typename EpilogueOutputOp::Params output_op;
                    int* semaphore;
                    int gemm_k_size;
                    int const* gather_A_indices;
                    int const* gather_B_indices;
                    int const* scatter_D_indices;


                    CUTLASS_HOST_DEVICE
                        Params()
                        : swizzle_log_tile(0)
                        , semaphore(0)
                        , gemm_k_size(0)
                    {
                    }

                    CUTLASS_HOST_DEVICE
                        Params(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape, const int gemm_k_size,
                            void* workspace = nullptr)
                        : problem_size(args.problem_size)
                        , group_size(args.group_size)
                        , grid_tiled_shape(grid_tiled_shape)
                        , swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape))
                        , params_A(args.ref_A.layout())
                        , ref_A(args.ref_A)
                        , params_B(args.ref_B.layout())
                        , ref_B(args.ref_B)
                        , params_scale(args.ref_scale.layout())
                        , ref_scale(args.ref_scale)
                        , ref_zero(args.ref_zero)
                        , params_C(args.ref_C.layout())
                        , ref_C(args.ref_C)
                        , params_D(args.ref_D.layout())
                        , ref_D(args.ref_D)
                        , output_op(args.output_op)
                        , semaphore(static_cast<int*>(workspace))
                        , gemm_k_size(gemm_k_size)
                        , gather_A_indices(args.gather_A_indices)
                        , gather_B_indices(args.gather_B_indices)
                        , scatter_D_indices(args.scatter_D_indices)
                    {
                    }
                };

                union SharedStorage
                {
                    typename Mma::SharedStorage main_loop;
                    typename Epilogue::SharedStorage epilogue;
                };


                CUTLASS_HOST_DEVICE
                    GemmFpAIntB() {}

                CUTLASS_HOST_DEVICE
                    static Status can_implement(Arguments const& args)
                {

                    static int const kAlignmentA
                        = (platform::is_same<typename Mma::IteratorA::Layout, layout::ColumnMajorInterleaved<32>>::value) ? 32
                        : (platform::is_same<typename Mma::IteratorA::Layout, layout::ColumnMajorInterleaved<64>>::value)
                        ? 64
                        : Mma::IteratorA::AccessType::kElements;
                    static int const kAlignmentB
                        = (platform::is_same<typename Mma::IteratorB::Layout, layout::RowMajorInterleaved<32>>::value) ? 32
                        : (platform::is_same<typename Mma::IteratorB::Layout, layout::RowMajorInterleaved<64>>::value)
                        ? 64
                        : Mma::IteratorB::AccessType::kElements;

                    static int const kAlignmentScale = Mma::IteratorScale::AccessType::kElements;

                    static int const kAlignmentC = (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                        layout::ColumnMajorInterleaved<32>>::value)
                        ? 32
                        : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                            layout::ColumnMajorInterleaved<64>>::value)
                        ? 64
                        : Epilogue::OutputTileIterator::kElementsPerAccess;

                    if (!TensorRef_aligned(args.ref_A, kAlignmentA))
                    {
                        return Status::kErrorMisalignedOperand;
                    }

                    if (!TensorRef_aligned(args.ref_B, kAlignmentB))
                    {
                        return Status::kErrorMisalignedOperand;
                    }

                    if (!TensorRef_aligned(args.ref_scale, kAlignmentScale))
                    {
                        return Status::kErrorMisalignedOperand;
                    }

                    if (!TensorRef_aligned(args.ref_zero, kAlignmentScale))
                    {
                        return Status::kErrorMisalignedOperand;
                    }

                    if (!TensorRef_aligned(args.ref_C, kAlignmentC))
                    {
                        return Status::kErrorMisalignedOperand;
                    }

                    if (!TensorRef_aligned(args.ref_D, kAlignmentC))
                    {
                        return Status::kErrorMisalignedOperand;
                    }

                    if (!args.ref_scale.good())
                    {
                        return Status::kErrorNotSupported;
                    }

                    if constexpr (hasZero(Mma::QuantOp))
                    {
                        if (!args.ref_zero.good())
                        {
                            return Status::kErrorNotSupported;
                        }
                    }
                    else
                    {
                        if (args.ref_zero.good())
                        {
                            return Status::kErrorNotSupported;
                        }
                    }

                    if constexpr (isFinegrained(Mma::QuantOp))
                    {
                        if (args.group_size != 64 && args.group_size != 128)
                        {
                            return Status::kErrorNotSupported;
                        }
                    }

                    return Status::kSuccess;
                }

                static size_t get_extra_workspace_size(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape)
                {

                    return 0;
                }

                template <bool B, typename dummy = void>
                struct KernelRunner
                {
                    CUTLASS_DEVICE
                        static void run_kernel(Params const& params, SharedStorage& shared_storage)
                    {
                        CUTLASS_NOT_IMPLEMENTED();
                    }
                };

                template <typename IteratorScale, WeightOnlyQuantOp op, std::enable_if_t<isFinegrained(op), bool> = true>
                CUTLASS_DEVICE static IteratorScale initialize_scale(typename IteratorScale::Params const& params,
                    typename IteratorScale::Pointer pointer_scale, typename IteratorScale::Pointer pointer_zero,
                    typename IteratorScale::TensorCoord extent, int thread_id,
                    typename IteratorScale::TensorCoord const& threadblock_offset, int group_size)
                {

                    return IteratorScale(params, pointer_scale, pointer_zero, extent, thread_id, threadblock_offset, group_size);
                }

                template <typename IteratorScale, WeightOnlyQuantOp op, std::enable_if_t<!isFinegrained(op), bool> = true>
                CUTLASS_DEVICE static IteratorScale initialize_scale(typename IteratorScale::Params const& params,
                    typename IteratorScale::Pointer pointer_scale, typename IteratorScale::Pointer pointer_zero,
                    typename IteratorScale::TensorCoord extent, int thread_id,
                    typename IteratorScale::TensorCoord const& threadblock_offset, int group_size)
                {

                    return IteratorScale(params, pointer_scale, extent, thread_id, threadblock_offset);
                }

                template <typename dummy>
                struct KernelRunner<true, dummy>
                {
                    CUTLASS_DEVICE
                        static void run_kernel(Params const& params, SharedStorage& shared_storage)
                    {
                        using LayoutB = typename Mma::IteratorB::Layout;
                        static_assert(platform::is_same<LayoutB, layout::RowMajor>::value && kInterleave == 1
                            || platform::is_same<LayoutB, layout::ColumnMajor>::value && kInterleave >= 1,
                            "B must be row major/col major OR col major interleaved.");

                        ThreadblockSwizzle threadblock_swizzle;

                        cutlass::gemm::GemmCoord threadblock_tile_offset
                            = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

                        if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m()
                            || params.grid_tiled_shape.n() <= threadblock_tile_offset.n())
                        {

                            return;
                        }

                        cutlass::MatrixCoord tb_offset_A{
                            threadblock_tile_offset.m() * Mma::Shape::kM,
                            threadblock_tile_offset.k() * params.gemm_k_size,
                        };

                        cutlass::MatrixCoord tb_offset_B{ threadblock_tile_offset.k() * params.gemm_k_size * kInterleave,
                            threadblock_tile_offset.n() * Mma::Shape::kN / kInterleave };

                        typename MatrixCoord::Index fg_row_offset = threadblock_tile_offset.k() * params.gemm_k_size / 64;
                        typename MatrixCoord::Index scale_row_offset = isFinegrained(Mma::QuantOp) ? fg_row_offset : 0;
                        cutlass::MatrixCoord tb_offset_scale{ scale_row_offset, threadblock_tile_offset.n() * Mma::Shape::kN };

                        int problem_size_k = min(params.problem_size.k(), (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

                        int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

                        int thread_idx = threadIdx.x;

                        typename Mma::IteratorA iterator_A(params.params_A, params.ref_A.data(),
                            { params.problem_size.m(), problem_size_k }, thread_idx, tb_offset_A, params.gather_A_indices);

                        typename Mma::IteratorB iterator_B(params.params_B, params.ref_B.data(),
                            { problem_size_k * kInterleave, params.problem_size.n() / kInterleave }, thread_idx, tb_offset_B,
                            params.gather_B_indices);

                        typename MatrixCoord::Index scale_row_extent = isFinegrained(Mma::QuantOp) ? problem_size_k / 64 : 1;
                        typename Mma::IteratorScale iterator_scale = initialize_scale<typename Mma::IteratorScale, Mma::QuantOp>(
                            params.params_scale, params.ref_scale.data(), params.ref_zero.data(),
                            { scale_row_extent, params.problem_size.n() }, thread_idx, tb_offset_scale, params.group_size);

                        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
                        int lane_idx = threadIdx.x % 32;

                        Mma mma(shared_storage.main_loop, params.group_size, thread_idx, warp_idx, lane_idx);

                        typename Mma::FragmentC accumulators;

                        accumulators.clear();

                        if (!kSplitKSerial || gemm_k_iterations > 0)
                        {
                            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_scale, accumulators);
                        }


                        EpilogueOutputOp output_op(params.output_op);


                        threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

                        MatrixCoord threadblock_offset(
                            threadblock_tile_offset.m() * Mma::Shape::kM, threadblock_tile_offset.n() * Mma::Shape::kN);

                        int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

                        Semaphore semaphore(params.semaphore + block_idx, thread_idx);

                        if (kSplitKSerial && params.grid_tiled_shape.k() > 1)
                        {

                            semaphore.fetch();

                            output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
                        }

                        typename Epilogue::OutputTileIterator iterator_C(params.params_C, params.ref_C.data(),
                            params.problem_size.mn(), thread_idx, threadblock_offset, params.scatter_D_indices);

                        typename Epilogue::OutputTileIterator iterator_D(params.params_D, params.ref_D.data(),
                            params.problem_size.mn(), thread_idx, threadblock_offset, params.scatter_D_indices);

                        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

                        if (kSplitKSerial && params.grid_tiled_shape.k() > 1)
                        {

                            if (threadblock_tile_offset.k())
                            {
                                iterator_C = iterator_D;
                            }

                            semaphore.wait(threadblock_tile_offset.k());
                        }

                        epilogue(output_op, iterator_D, accumulators, iterator_C);


                        if (kSplitKSerial && params.grid_tiled_shape.k() > 1)
                        {

                            int lock = 0;
                            if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1)
                            {

                                lock = 0;
                            }
                            else
                            {
                                lock = threadblock_tile_offset.k() + 1;
                            }

                            semaphore.release(lock);
                        }
                    }
                };

                CUTLASS_DEVICE
                    void operator()(Params const& params, SharedStorage& shared_storage)
                {
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
                    static constexpr bool compile_needed = platform::is_same<KernelArch, arch::Sm70>::value;
                    KernelRunner<compile_needed>::run_kernel(params, shared_storage);
#elif (__CUDA_ARCH__ >= 750) && (__CUDA_ARCH__ < 800)
                    static constexpr bool compile_needed = platform::is_same<KernelArch, arch::Sm75>::value;
                    KernelRunner<compile_needed>::run_kernel(params, shared_storage);
#elif (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ <= 900)
                    static constexpr bool compile_needed = platform::is_same<KernelArch, arch::Sm80>::value;
                    KernelRunner<compile_needed>::run_kernel(params, shared_storage);
#else
                    static_assert(
                        false, "Invalid architecture being compiled. Only Volta+ supported in weight-only quantization kernels.");
#endif
#else
                    CUTLASS_NOT_IMPLEMENTED();
#endif
                }
            };


        }
    }
}