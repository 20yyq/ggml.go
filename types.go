// @@
// @ Author       : Eacher
// @ Date         : 2026-05-25 14:30:51
// @ LastEditTime : 2026-05-25 20:29:59
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package ggmlgo

type GGML_STATUS int32

const (
	GGML_STATUS_ALLOC_FAILED GGML_STATUS = iota - 2
	GGML_STATUS_FAILED
	GGML_STATUS_SUCCESS
	GGML_STATUS_ABORTED
)

// NOTE: always add types at the end of the enum to keep backward compatibility
type GGML_TYPE uint32

const (
	GGML_TYPE_F32 GGML_TYPE = iota
	GGML_TYPE_F16
	GGML_TYPE_Q4_0
	GGML_TYPE_Q4_1
	GGML_TYPE_Q4_2
	GGML_TYPE_Q4_3
	GGML_TYPE_Q5_0
	GGML_TYPE_Q5_1
	GGML_TYPE_Q8_0
	GGML_TYPE_Q8_1
	GGML_TYPE_Q2_K
	GGML_TYPE_Q3_K
	GGML_TYPE_Q4_K
	GGML_TYPE_Q5_K
	GGML_TYPE_Q6_K
	GGML_TYPE_Q8_K
	GGML_TYPE_IQ2_XXS
	GGML_TYPE_IQ2_XS
	GGML_TYPE_IQ3_XXS
	GGML_TYPE_IQ1_S
	GGML_TYPE_IQ4_NL
	GGML_TYPE_IQ3_S
	GGML_TYPE_IQ2_S
	GGML_TYPE_IQ4_XS
	GGML_TYPE_I8
	GGML_TYPE_I16
	GGML_TYPE_I32
	GGML_TYPE_I64
	GGML_TYPE_F64
	GGML_TYPE_IQ1_M
	GGML_TYPE_BF16
	GGML_TYPE_Q4_0_4_4
	GGML_TYPE_Q4_0_4_8
	GGML_TYPE_Q4_0_8_8
	GGML_TYPE_TQ1_0
	GGML_TYPE_TQ2_0
	GGML_TYPE_IQ4_NL_4_4
	GGML_TYPE_IQ4_NL_4_8
	GGML_TYPE_IQ4_NL_8_8
	GGML_TYPE_MXFP4 // MXFP4 (1 block)
	GGML_TYPE_NVFP4 // NVFP4 (4 blocks, E4M3 scale)
	GGML_TYPE_Q1_0
	GGML_TYPE_COUNT
)

type GGML_PREC int32

const (
	GGML_PREC_DEFAULT GGML_PREC = iota // stored as ggml_tensor.op_params, 0 by default
	GGML_PREC_F32               = 10
)

// op hint
type GGML_OP_HINT int32

const (
	GGML_HINT_NONE GGML_OP_HINT = iota // stored as ggml_tensor.op_params, 0 by default
	GGML_HINT_SRC0_IS_HADAMARD
)

// model file types
type GGML_FTYPE int32

const (
	GGML_FTYPE_UNKNOWN GGML_FTYPE = iota - 1 // stored as ggml_tensor.op_params, 0 by default
	GGML_FTYPE_ALL_F32
	GGML_FTYPE_MOSTLY_F16                      // except 1d tensors
	GGML_FTYPE_MOSTLY_Q4_0                     // except 1d tensors
	GGML_FTYPE_MOSTLY_Q4_1                     // except 1d tensors
	GGML_FTYPE_MOSTLY_Q4_1_SOME_F16            // tok_embeddings.weight and output.weight are F16
	GGML_FTYPE_MOSTLY_Q8_0          = iota + 1 // except 1d tensors
	GGML_FTYPE_MOSTLY_Q5_0                     // except 1d tensors
	GGML_FTYPE_MOSTLY_Q5_1                     // except 1d tensors
	GGML_FTYPE_MOSTLY_Q2_K                     // except 1d tensors
	GGML_FTYPE_MOSTLY_Q3_K                     // except 1d tensors
	GGML_FTYPE_MOSTLY_Q4_K                     // except 1d tensors
	GGML_FTYPE_MOSTLY_Q5_K                     // except 1d tensors
	GGML_FTYPE_MOSTLY_Q6_K                     // except 1d tensors
	GGML_FTYPE_MOSTLY_IQ2_XXS                  // except 1d tensors
	GGML_FTYPE_MOSTLY_IQ2_XS                   // except 1d tensors
	GGML_FTYPE_MOSTLY_IQ3_XXS                  // except 1d tensors
	GGML_FTYPE_MOSTLY_IQ1_S                    // except 1d tensors
	GGML_FTYPE_MOSTLY_IQ4_NL                   // except 1d tensors
	GGML_FTYPE_MOSTLY_IQ3_S                    // except 1d tensors
	GGML_FTYPE_MOSTLY_IQ2_S                    // except 1d tensors
	GGML_FTYPE_MOSTLY_IQ4_XS                   // except 1d tensors
	GGML_FTYPE_MOSTLY_IQ1_M                    // except 1d tensors
	GGML_FTYPE_MOSTLY_BF16                     // except 1d tensors
	GGML_FTYPE_MOSTLY_MXFP4                    // except 1d tensors
	GGML_FTYPE_MOSTLY_NVFP4                    // except 1d tensors
	GGML_FTYPE_MOSTLY_Q1_0                     // except 1d tensors
)

// available tensor operations:
type GGML_OP uint32

const (
	GGML_OP_NONE GGML_OP = iota // stored as ggml_tensor.op_params, 0 by default
	GGML_OP_DUP
	GGML_OP_ADD
	GGML_OP_ADD_ID
	GGML_OP_ADD1
	GGML_OP_ACC
	GGML_OP_SUB
	GGML_OP_MUL
	GGML_OP_DIV
	GGML_OP_SQR
	GGML_OP_SQRT
	GGML_OP_LOG
	GGML_OP_SIN
	GGML_OP_COS
	GGML_OP_SUM
	GGML_OP_SUM_ROWS
	GGML_OP_CUMSUM
	GGML_OP_MEAN
	GGML_OP_ARGMAX
	GGML_OP_COUNT_EQUAL
	GGML_OP_REPEAT
	GGML_OP_REPEAT_BACK
	GGML_OP_CONCAT
	GGML_OP_SILU_BACK
	GGML_OP_NORM // normalize
	GGML_OP_RMS_NORM
	GGML_OP_RMS_NORM_BACK
	GGML_OP_GROUP_NORM
	GGML_OP_L2_NORM

	GGML_OP_MUL_MAT
	GGML_OP_MUL_MAT_ID
	GGML_OP_OUT_PROD

	GGML_OP_SCALE
	GGML_OP_SET
	GGML_OP_CPY
	GGML_OP_CONT
	GGML_OP_RESHAPE
	GGML_OP_VIEW
	GGML_OP_PERMUTE
	GGML_OP_TRANSPOSE
	GGML_OP_GET_ROWS
	GGML_OP_GET_ROWS_BACK
	GGML_OP_SET_ROWS
	GGML_OP_DIAG
	GGML_OP_DIAG_MASK_INF
	GGML_OP_DIAG_MASK_ZERO
	GGML_OP_SOFT_MAX
	GGML_OP_SOFT_MAX_BACK
	GGML_OP_ROPE
	GGML_OP_ROPE_BACK
	GGML_OP_CLAMP
	GGML_OP_CONV_TRANSPOSE_1D
	GGML_OP_IM2COL
	GGML_OP_IM2COL_BACK
	GGML_OP_IM2COL_3D
	GGML_OP_CONV_2D
	GGML_OP_CONV_3D
	GGML_OP_CONV_2D_DW
	GGML_OP_CONV_TRANSPOSE_2D
	GGML_OP_POOL_1D
	GGML_OP_POOL_2D
	GGML_OP_POOL_2D_BACK
	GGML_OP_UPSCALE
	GGML_OP_PAD
	GGML_OP_PAD_REFLECT_1D
	GGML_OP_ROLL
	GGML_OP_ARANGE
	GGML_OP_TIMESTEP_EMBEDDING
	GGML_OP_ARGSORT
	GGML_OP_TOP_K
	GGML_OP_LEAKY_RELU
	GGML_OP_TRI
	GGML_OP_FILL

	GGML_OP_FLASH_ATTN_EXT
	GGML_OP_FLASH_ATTN_BACK
	GGML_OP_SSM_CONV
	GGML_OP_SSM_SCAN
	GGML_OP_WIN_PART
	GGML_OP_WIN_UNPART
	GGML_OP_GET_REL_POS
	GGML_OP_ADD_REL_POS
	GGML_OP_RWKV_WKV6
	GGML_OP_GATED_LINEAR_ATTN
	GGML_OP_RWKV_WKV7
	GGML_OP_SOLVE_TRI
	GGML_OP_GATED_DELTA_NET

	GGML_OP_UNARY

	GGML_OP_MAP_CUSTOM1
	GGML_OP_MAP_CUSTOM2
	GGML_OP_MAP_CUSTOM3

	GGML_OP_CUSTOM

	GGML_OP_CROSS_ENTROPY_LOSS
	GGML_OP_CROSS_ENTROPY_LOSS_BACK
	GGML_OP_OPT_STEP_ADAMW
	GGML_OP_OPT_STEP_SGD

	GGML_OP_GLU

	GGML_OP_COUNT
)

// ggml_unary_op
type GGML_UNARY_OP uint32

const (
	GGML_UNARY_OP_ABS GGML_UNARY_OP = iota // stored as ggml_tensor.op_params, 0 by default
	GGML_UNARY_OP_SGN
	GGML_UNARY_OP_NEG
	GGML_UNARY_OP_STEP
	GGML_UNARY_OP_TANH
	GGML_UNARY_OP_ELU
	GGML_UNARY_OP_RELU
	GGML_UNARY_OP_SIGMOID
	GGML_UNARY_OP_GELU
	GGML_UNARY_OP_GELU_QUICK
	GGML_UNARY_OP_SILU
	GGML_UNARY_OP_HARDSWISH
	GGML_UNARY_OP_HARDSIGMOID
	GGML_UNARY_OP_EXP
	GGML_UNARY_OP_EXPM1
	GGML_UNARY_OP_SOFTPLUS
	GGML_UNARY_OP_GELU_ERF
	GGML_UNARY_OP_XIELU
	GGML_UNARY_OP_FLOOR
	GGML_UNARY_OP_CEIL
	GGML_UNARY_OP_ROUND
	GGML_UNARY_OP_TRUNC

	GGML_UNARY_OP_COUNT
)

type GGML_GLU_OP uint32

const (
	GGML_GLU_OP_REGLU GGML_GLU_OP = iota // stored as ggml_tensor.op_params, 0 by default
	GGML_GLU_OP_GEGLU
	GGML_GLU_OP_SWIGLU
	GGML_GLU_OP_SWIGLU_OAI
	GGML_GLU_OP_GEGLU_ERF
	GGML_GLU_OP_GEGLU_QUICK
	GGML_GLU_OP_COUNT
)

type GGML_OBJECT_TYPE int32

const (
	GGML_OBJECT_TYPE_TENSOR GGML_OBJECT_TYPE = iota // stored as ggml_tensor.op_params, 0 by default
	GGML_OBJECT_TYPE_GRAPH
	GGML_OBJECT_TYPE_WORK_BUFFER
)

type GGML_LOG_LEVEL int32

const (
	GGML_LOG_LEVEL_NONE GGML_LOG_LEVEL = iota // stored as ggml_tensor.op_params, 0 by default
	GGML_LOG_LEVEL_DEBUG
	GGML_LOG_LEVEL_INFO
	GGML_LOG_LEVEL_WARN
	GGML_LOG_LEVEL_ERROR
	GGML_LOG_LEVEL_CONT // continue previous log
)

// this tensor...
type GGML_TENSOR_FLAG int32

const (
	GGML_TENSOR_FLAG_INPUT   GGML_TENSOR_FLAG = iota + 1 // ...is an input for the GGML compute graph
	GGML_TENSOR_FLAG_OUTPUT                   = 2        // ...is an output for the GGML compute graph
	GGML_TENSOR_FLAG_PARAM                    = 4        // ...contains trainable parameters
	GGML_TENSOR_FLAG_LOSS                     = 8        // ...defines loss for numerical optimization (multiple loss tensors add up)
	GGML_TENSOR_FLAG_COMPUTE                  = 16       // ...must be computed
)

type GGML_TRI_TYPE int32

const (
	GGML_TRI_TYPE_UPPER_DIAG GGML_TRI_TYPE = iota // stored as ggml_tensor.op_params, 0 by default
	GGML_TRI_TYPE_UPPER
	GGML_TRI_TYPE_LOWER_DIAG
	GGML_TRI_TYPE_LOWER
)
