
namespace bitfusion
{
namespace kernels
{
// clang-format off
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_80_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_86_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_2_nqpkv_8_sm_89_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_90_cubin[];
extern unsigned long long xqa_kernel_dt_fp16_d_128_beam_1_kvt_2_nqpkv_8_sm_90_cubin[];

extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_80_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_86_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_2_nqpkv_8_sm_89_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_90_cubin_len;
extern uint32_t xqa_kernel_dt_fp16_d_128_beam_1_kvt_2_nqpkv_8_sm_90_cubin_len;


static const struct XQAKernelMetaInfo
{
    Data_type mDataType;
    Data_type mKVDataType;
    unsigned int mHeadDim;
    unsigned int mBeamWidth;
    unsigned int mNumQHeadsOverKV;
    unsigned int mSM;
    const unsigned long long* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
} sXqaKernelMetaInfo[] = {
{ DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, kSM_80, xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_80_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_80_cubin_len, "kernel_mha"},
{ DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, kSM_80, xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_80_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_80_cubin_len, "kernel_mha"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, kSM_86, xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_86_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_86_cubin_len, "kernel_mha"},
{ DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, kSM_86, xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_86_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_86_cubin_len, "kernel_mha"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, kSM_89, xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_89_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_89_cubin_len, "kernel_mha"},
{ DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, kSM_89, xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_89_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_89_cubin_len, "kernel_mha"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, kSM_89, xqa_kernel_dt_fp16_d_128_beam_1_kvt_2_nqpkv_8_sm_89_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_2_nqpkv_8_sm_89_cubin_len, "kernel_mha"},
{ DATA_TYPE_FP16, DATA_TYPE_FP16, 128, 1, 8, kSM_90, xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_90_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_0_nqpkv_8_sm_90_cubin_len, "kernel_mha"},
{ DATA_TYPE_FP16, DATA_TYPE_INT8, 128, 1, 8, kSM_90, xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_90_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_1_nqpkv_8_sm_90_cubin_len, "kernel_mha"},
{ DATA_TYPE_FP16, DATA_TYPE_E4M3, 128, 1, 8, kSM_90, xqa_kernel_dt_fp16_d_128_beam_1_kvt_2_nqpkv_8_sm_90_cubin, xqa_kernel_dt_fp16_d_128_beam_1_kvt_2_nqpkv_8_sm_90_cubin_len, "kernel_mha"}
};

// clang-format on
} // namespace kernels
} // namespace bitfusion
