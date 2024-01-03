#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cutlass/platform/platform.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/linear_combination_hardswish.h"
#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_relu0.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"

#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/reduction_op.h"

#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_affine.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_strided_dgrad.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator_mixed.h"
#include "cutlass/epilogue/warp/fragment_iterator_complex_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/interleaved_epilogue.h"

#include "cutlass/layout/permute.h"


namespace cutlass
{
    namespace epilogue
    {
        namespace threadblock
        {


            namespace detail
            {

                template <typename ThreadblockShape, typename WarpShape, typename InstructionShape, typename ThreadMap>
                struct DefaultIteratorsTensorOp<cutlass::bfloat16_t, int32_t, 8, ThreadblockShape, WarpShape, InstructionShape,
                    ThreadMap>
                {
                    using WarpTileIterator
                        = cutlass::epilogue::warp::TileIteratorTensorOpMixed<WarpShape, InstructionShape, int32_t, 32, 16, 8, 8>;

                    using SharedLoadIterator
                        = cutlass::epilogue::threadblock::SharedLoadIteratorMixed<ThreadMap, int32_t, 32, 16, 8, 8>;

                    static int const kFragmentsPerIteration = 2;
                };


            }


            template <typename ThreadMap_
            >
            class SharedLoadIteratorMixed<ThreadMap_, int32_t, 32, 16, 8, 8>
            {
            public:
                using ThreadMap = ThreadMap_;
                using Shape = typename ThreadMap::Shape;

                using Element = int32_t;

                using Layout = layout::RowMajor;
                using TensorRef = TensorRef<Element, Layout>;
                using ConstTensorRef = typename TensorRef::ConstTensorRef;

                using Index = typename Layout::Index;
                using LongIndex = typename Layout::LongIndex;
                using TensorCoord = MatrixCoord;

                static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

                static int const kAlignment = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value / 8;

                static int const kThreads = ThreadMap::kThreads;

                using Fragment = Array<Element,
                    ThreadMap::Iterations::kColumn* ThreadMap::Iterations::kRow* ThreadMap::Iterations::kGroup
                    * ThreadMap::Iterations::kCluster* ThreadMap::kElementsPerAccess>;

                using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess, kAlignment>;

                using LoadType = AlignedArray<Element, const_min(128 / sizeof_bits<Element>::value, ThreadMap::kElementsPerAccess),
                    const_min(16, kAlignment)>;

                static int const kLoadsPerAccess = AccessType::kElements / LoadType::kElements;

            private:

                LoadType const* pointers_[kLoadsPerAccess];

                int stride_;

            public:

                CUTLASS_DEVICE
                    SharedLoadIteratorMixed(TensorRef ref, int thread_idx)
                    : stride_((ref.stride(0) / LoadType::kElements))
                {

                    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

                    CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kLoadsPerAccess; ++i)
                        {
                            pointers_[i] = reinterpret_cast<LoadType const*>(ref.data());

                            int col_idx = (thread_offset.column() / kElementsPerAccess) * kLoadsPerAccess;
                            int bank_offset = (col_idx * int(sizeof(LoadType)) / 128) % kLoadsPerAccess;

                            col_idx += (bank_offset + i) % kLoadsPerAccess;

                            pointers_[i] += thread_offset.row() * stride_ + col_idx;
                        }
                }

                CUTLASS_HOST_DEVICE
                    void add_pointer_offset(LongIndex pointer_offset)
                {
                    CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kLoadsPerAccess; ++i)
                        {
                            pointers_[i] += pointer_offset / LoadType::kElements;
                        }
                }

                CUTLASS_DEVICE
                    void add_tile_offset(TensorCoord const& offset)
                {
                    CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kLoadsPerAccess; ++i)
                        {
                            pointers_[i]
                                += offset.row() * Shape::kRow * stride_ + offset.column() * Shape::kColumn / LoadType::kElements;
                        }
                }

                CUTLASS_DEVICE
                    void load_with_pointer_offset(Fragment& frag, Index pointer_offset) const
                {

                    CUTLASS_PRAGMA_UNROLL
                        for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster)
                        {

                            CUTLASS_PRAGMA_UNROLL
                                for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group)
                                {

                                    CUTLASS_PRAGMA_UNROLL
                                        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row)
                                        {

                                            int row_ptr_offset = row * ThreadMap::Delta::kRow * stride_
                                                + group * ThreadMap::Delta::kGroup * stride_ + cluster * ThreadMap::Delta::kCluster * stride_
                                                + pointer_offset / LoadType::kElements;

                                            int frag_row_idx
                                                = (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

                                            LoadType* frag_ptr = reinterpret_cast<LoadType*>(&frag);

                                            CUTLASS_PRAGMA_UNROLL
                                                for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column)
                                                {

                                                    int frag_idx = frag_row_idx * ThreadMap::Iterations::kColumn + column;

                                                    CUTLASS_PRAGMA_UNROLL
                                                        for (int v = 0; v < kLoadsPerAccess; ++v)
                                                        {

                                                            int vector_idx
                                                                = (column * ThreadMap::Delta::kColumn / kElementsPerAccess * kLoadsPerAccess);

                                                            LoadType const* memory_pointer = pointers_[v] + row_ptr_offset;

                                                            frag_ptr[frag_idx * kLoadsPerAccess + v] = memory_pointer[vector_idx];
                                                        }
                                                }
                                        }
                                }
                        }
                }

                CUTLASS_DEVICE
                    void load(Fragment& frag) const
                {

                    load_with_pointer_offset(frag, 0);
                }
            };


        }
    }
}