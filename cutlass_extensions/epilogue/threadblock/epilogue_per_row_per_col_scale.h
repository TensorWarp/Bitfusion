#pragma once


#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_conversion.h"
#include "../../../../common/quantization.h"

namespace tk = bitfusion::common;

namespace cutlass
{
    namespace epilogue
    {
        namespace threadblock
        {

            template <typename ThreadblockShape_, int ThreadCount, typename ScaleTileIterator_, typename OutputTileIterator_,
                typename ElementAccumulator_, typename ElementCompute_, typename ElementwiseFunctor_, bool UseMasking_ = false>
            class EpilogueVisitorPerRowPerCol
            {
            public:
                using ThreadblockShape = ThreadblockShape_;
                static int const kThreadCount = ThreadCount;

                using ScaleTileIterator = ScaleTileIterator_;
                using OutputTileIterator = OutputTileIterator_;
                using ElementwiseFunctor = ElementwiseFunctor_;

                static int const kIterations = OutputTileIterator::kIterations;
                static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

                using ElementOutput = typename OutputTileIterator::Element;
                using LayoutOutput = cutlass::layout::RowMajor;
                using ElementAccumulator = ElementAccumulator_;

                using AlphaScaleElementType = typename ScaleTileIterator::Element;

                using ElementCompute = ElementCompute_;
                using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
                using ComputeFragment = Array<ElementCompute_, kElementsPerAccess>;
                using OutputVector = Array<ElementOutput, kElementsPerAccess>;

                static int const kThreadsPerRow = OutputTileIterator::ThreadMap::Detail::kAccessWidth;
                static bool const kHasMultiStepsInRow = (OutputTileIterator::ThreadMap::Iterations::kColumn > 1);

                struct Arguments
                {

                    typename ElementwiseFunctor::Params elementwise;
                    int64_t batch_stride_alpha;
                    int64_t batch_stride_C;
                    int64_t batch_stride_D;

                    Arguments()
                        : batch_stride_alpha(0)
                        , batch_stride_C(0)
                        , batch_stride_D(0)
                    {
                    }

                    Arguments(typename ElementwiseFunctor::Params elementwise_)
                        : elementwise(elementwise_)
                        , batch_stride_alpha(0)
                        , batch_stride_C(0)
                        , batch_stride_D(0)
                    {
                    }

                    Arguments(typename ElementwiseFunctor::Params elementwise_, int64_t batch_stride_alpha_,
                        int64_t batch_stride_C_, int64_t batch_stride_D_)
                        : elementwise(elementwise_)
                        , batch_stride_alpha(batch_stride_alpha_)
                        , batch_stride_C(batch_stride_C_)
                        , batch_stride_D(batch_stride_D_)
                    {
                    }
                };

                struct Params
                {

                    typename ElementwiseFunctor::Params elementwise;
                    int64_t batch_stride_alpha;
                    int64_t batch_stride_C;
                    int64_t batch_stride_D;

                    CUTLASS_HOST_DEVICE
                        Params() {}

                    CUTLASS_HOST_DEVICE
                        Params(Arguments const& args)
                        : elementwise(args.elementwise)
                        , batch_stride_alpha(args.batch_stride_alpha)
                        , batch_stride_C(args.batch_stride_C)
                        , batch_stride_D(args.batch_stride_D)
                    {
                    }
                };

                struct SharedStorage
                {
                };

            private:
                Params const& params_;
                SharedStorage& shared_storage_;
                MatrixCoord extent_;
                MatrixCoord extent_real_;
                ElementwiseFunctor elementwise_;

                const bool per_token_quant_;
                const bool per_channel_quant_;

                AlphaScaleElementType* ptr_alpha_row_;
                AlphaScaleElementType* ptr_alpha_col_;
                ScaleTileIterator iterator_alpha_col_;
                OutputTileIterator iterator_C_;
                OutputTileIterator iterator_D_;

                AlphaScaleElementType element_alpha_row_ = 1.0f;
                AlphaScaleElementType element_alpha_col_ = 1.0f;
                typename ScaleTileIterator::Fragment fragment_alpha_col_;
                typename OutputTileIterator::Fragment fragment_C_;
                typename OutputTileIterator::Fragment fragment_D_;

                ElementAccumulator beta_;

                int column_offset_;

                MatrixCoord thread_offset_;

            public:
                CUTLASS_DEVICE
                    EpilogueVisitorPerRowPerCol(Params const& params, SharedStorage& shared_storage,
                        cutlass::MatrixCoord const& problem_size, int thread_idx, int warp_idx, int lane_idx,
                        typename ScaleTileIterator::Params params_alpha_col, typename OutputTileIterator::Params params_C,
                        typename OutputTileIterator::Params params_D, tk::QuantMode quant_option, AlphaScaleElementType* ptr_alpha_row,
                        AlphaScaleElementType* ptr_alpha_col, typename OutputTileIterator::Element* ptr_C,
                        typename OutputTileIterator::Element* ptr_D,
                        cutlass::MatrixCoord const& threadblock_offset = cutlass::MatrixCoord(0, 0), int column_offset = 0,
                        cutlass::MatrixCoord const& problem_size_real = cutlass::MatrixCoord(0, 0))
                    : params_(params)
                    , shared_storage_(shared_storage)
                    , extent_(problem_size)
                    , elementwise_(params.elementwise)
                    , per_token_quant_(quant_option.hasPerTokenScaling())
                    , per_channel_quant_(quant_option.hasPerChannelScaling())
                    , ptr_alpha_row_(ptr_alpha_row)
                    , ptr_alpha_col_(ptr_alpha_col)
                    , iterator_alpha_col_(params_alpha_col, ptr_alpha_col, problem_size, thread_idx, threadblock_offset)
                    , iterator_C_(params_C, ptr_C, problem_size, thread_idx, threadblock_offset)
                    , iterator_D_(params_D, ptr_D, problem_size, thread_idx, threadblock_offset)
                    , extent_real_(problem_size_real)
                {
                    beta_ = (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr : params.elementwise.beta);

                    if (beta_ == ElementAccumulator())
                    {
                        iterator_C_.clear_mask();
                    }

                    if (!per_channel_quant_ && (ptr_alpha_col_ != nullptr))
                    {
                        element_alpha_col_ = *ptr_alpha_col_;
                    }

                    if (!per_token_quant_ && (ptr_alpha_row_ != nullptr))
                    {
                        element_alpha_row_ = *ptr_alpha_row_;
                    }
                }

                CUTLASS_DEVICE
                    void set_k_partition(int split_k_index,
                        int split_k_slices)
                {
                }

                CUTLASS_DEVICE
                    void set_batch_index(int batch_idx)
                {
                    iterator_alpha_col_.add_pointer_offset(batch_idx * params_.batch_stride_alpha);
                    iterator_C_.add_pointer_offset(batch_idx * params_.batch_stride_C);
                    iterator_D_.add_pointer_offset(batch_idx * params_.batch_stride_D);
                }

                CUTLASS_DEVICE
                    void begin_epilogue()
                {
                    if (per_channel_quant_)
                    {
                        iterator_alpha_col_.load(fragment_alpha_col_);
                    }
                }

                CUTLASS_DEVICE
                    void begin_step(int step_idx)
                {
                    fragment_D_.clear();
                    fragment_C_.clear();

                    if (elementwise_.kScale != cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling)
                    {
                        iterator_C_.load(fragment_C_);
                        ++iterator_C_;
                    }
                }

                CUTLASS_DEVICE
                    void begin_row(int row_idx)
                {
                    if (per_token_quant_)
                    {
                        int thread_offset_row
                            = iterator_D_.thread_start_row() + OutputTileIterator::ThreadMap::iteration_offset(row_idx).row();

                        arch::global_load<AlphaScaleElementType, sizeof(AlphaScaleElementType)>(
                            element_alpha_row_, ptr_alpha_row_ + thread_offset_row, thread_offset_row < extent_.row());
                    }
                }

                CUTLASS_DEVICE
                    void visit(int iter_idx, int row_idx, int column_idx, int frag_idx, AccumulatorFragment const& accum)
                {

                    NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess> source_converter;

                    ComputeFragment result = source_converter(accum);
                    if (per_channel_quant_)
                    {
                        ComputeFragment alpha_col = reinterpret_cast<ComputeFragment*>(&fragment_alpha_col_)[column_idx];
                        result = per_token_channel_scale_accumulator_(result, alpha_col, element_alpha_row_);
                    }
                    else
                    {
                        result = per_token_scale_accumulator_(result, element_alpha_col_, element_alpha_row_);
                    }

                    NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess> output_converter;
                    OutputVector& output = reinterpret_cast<OutputVector*>(&fragment_D_)[frag_idx];
                    output = output_converter(result);
                }

                CUTLASS_DEVICE
                    void end_row(int row_idx) {}

                CUTLASS_DEVICE
                    void end_step(int step_idx)
                {

                    iterator_D_.store(fragment_D_);
                    ++iterator_D_;
                }

                CUTLASS_DEVICE
                    void end_epilogue() {}

            private:
                CUTLASS_DEVICE
                    ComputeFragment per_token_channel_scale_accumulator_(
                        ComputeFragment const& accum, ComputeFragment const& scale_col, AlphaScaleElementType const& scale_row)
                {

                    ComputeFragment result;
                    CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < ComputeFragment::kElements; ++i)
                        {
                            result[i] = accum[i] * (scale_col[i] * scale_row);
                        }

                    return result;
                }

                CUTLASS_DEVICE
                    ComputeFragment per_token_scale_accumulator_(
                        ComputeFragment const& accum, AlphaScaleElementType const& scale_col, AlphaScaleElementType const& scale_row)
                {

                    ComputeFragment result;
                    CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < ComputeFragment::kElements; ++i)
                        {
                            result[i] = accum[i] * (scale_col * scale_row);
                        }

                    return result;
                }
            };

        }
    }
}