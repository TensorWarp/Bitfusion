#pragma once

#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator_params.h"



namespace cutlass
{
    namespace transform
    {
        namespace threadblock
        {


            template <typename Shape, typename Element, typename Layout, int AdvanceRank, int Alignment>
            class FineGrainedScaleZeroIterator;

            template <typename Shape_, typename Element_, int Alignment_>
            class FineGrainedScaleZeroIterator<Shape_, Element_, layout::RowMajor, 0, Alignment_>
            {
            public:
                using Shape = Shape_;
                using Element = Element_;
                using Layout = layout::RowMajor;
                static int const kAdvanceRank = 0;
                static int const kAlignment = Alignment_;

                static int const kAccessesPerVector = 1;

                int row_groupsize64_;
                int group_size_;

                using Index = typename Layout::Index;
                using LongIndex = typename Layout::LongIndex;

                using TensorRef = TensorRef<Element, Layout>;
                using TensorView = TensorView<Element, Layout>;
                using TensorCoord = typename Layout::TensorCoord;
                using Pointer = Element*;
                using NonConstPointer = typename platform::remove_const<Element>::type*;

                using AccessType = AlignedArray<Element, kAlignment>;

                struct Params
                {
                    LongIndex stride_ = 0;

                    LongIndex inc_advance_ = 0;

                    CUTLASS_HOST_DEVICE
                        Params() {}

                    CUTLASS_HOST_DEVICE
                        Params(Layout const& layout)
                        : stride_(layout.stride(0))
                    {
                        inc_advance_ = Shape::kRow * stride_ * sizeof_bits<Element>::value / 8;
                    }
                };

            private:
                using BytePointer = char*;

            private:

                Params const params_;

                BytePointer pointer_scale_;
                BytePointer pointer_zero_;

                bool is_valid_ = false;

            public:
                CUTLASS_DEVICE
                    FineGrainedScaleZeroIterator(
                        Params const& params,
                        Pointer pointer_scale,
                        Pointer pointer_zero,
                        TensorCoord extent,
                        int thread_id,
                        TensorCoord const& threadblock_offset,
                        int group_size)
                    : params_(params)
                    , pointer_scale_(reinterpret_cast<BytePointer>(const_cast<NonConstPointer>(pointer_scale)))
                    , pointer_zero_(reinterpret_cast<BytePointer>(const_cast<NonConstPointer>(pointer_zero)))
                {
                    row_groupsize64_ = threadblock_offset.row();
                    group_size_ = group_size;

                    const LongIndex tb_row_byte_offset
                        = threadblock_offset.row() / (group_size / 64) * params_.stride_ * sizeof_bits<Element>::value / 8;
                    const LongIndex tb_col_byte_offset = threadblock_offset.column() * sizeof_bits<Element>::value / 8;
                    pointer_scale_ += (tb_row_byte_offset + tb_col_byte_offset);

                    if (pointer_zero_ != nullptr)
                    {
                        pointer_zero_ += (tb_row_byte_offset + tb_col_byte_offset);
                    }

                    static constexpr int THREADS_PER_ROW = Shape::kColumn / kAlignment;

                    const int thread_row = thread_id / THREADS_PER_ROW;
                    const int thread_col = thread_id % THREADS_PER_ROW;

                    const LongIndex thread_row_byte_offset = thread_row * params_.stride_ * sizeof_bits<Element>::value / 8;
                    const LongIndex thread_col_byte_offset = thread_col * kAlignment * sizeof_bits<Element>::value / 8;
                    pointer_scale_ += (thread_row_byte_offset + thread_col_byte_offset);
                    if (pointer_zero_ != nullptr)
                    {
                        pointer_zero_ += (thread_row_byte_offset + thread_col_byte_offset);
                    }

                    const int global_row = threadblock_offset.row() + thread_row;
                    const int global_col = threadblock_offset.column() + thread_col * kAlignment;

                    const bool row_in_bounds = global_row < extent.row() && thread_row < Shape::kRow;
                    const bool col_in_bounds = global_col < extent.column();

                    is_valid_ = row_in_bounds && col_in_bounds;
                }

                CUTLASS_HOST_DEVICE FineGrainedScaleZeroIterator(Params const& params,
                    Pointer pointer_scale,
                    Pointer pointer_zero,
                    TensorCoord extent,
                    int thread_id,
                    int group_size)
                    : FineGrainedScaleZeroIterator(
                        params, pointer_scale, pointer_zero, extent, thread_id, make_Coord(0, 0), group_size)
                {
                }

                CUTLASS_DEVICE
                    void add_tile_offset(TensorCoord const& tile_offset)
                {
                    const LongIndex row_byte_offset = tile_offset.row() * params_.inc_advance_;
                    const LongIndex col_byte_offset = tile_offset.column() * Shape::kColumn * sizeof_bits<Element>::value / 8;
                    pointer_scale_ += row_byte_offset + col_byte_offset;
                    if (pointer_zero_ != nullptr)
                    {
                        pointer_zero_ += row_byte_offset + col_byte_offset;
                    }
                }

                CUTLASS_HOST_DEVICE void clear_mask(bool enable = true)
                {
                    is_valid_ &= (!enable);
                }

                CUTLASS_HOST_DEVICE
                    bool valid() const
                {
                    return is_valid_;
                }

                CUTLASS_HOST_DEVICE
                    AccessType* get_scale() const
                {
                    return reinterpret_cast<AccessType*>(pointer_scale_);
                }

                CUTLASS_HOST_DEVICE
                    AccessType* get_zero() const
                {
                    return reinterpret_cast<AccessType*>(pointer_zero_);
                }
            };

        }
    }
}