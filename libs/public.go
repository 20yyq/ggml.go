// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 09:42:34
// @ LastEditTime : 2026-06-30 10:42:42
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

// #include "expand.h"
import "C"
import (
	"errors"
	"sync/atomic"

	ggmlgo "ggml.go"
)

var is_init atomic.Bool

type InitParams struct {
	Numa bool
}

func Init(p InitParams) error {
	if is_init.Swap(true) {
		return errors.New("is init")
	}
	backend_init(p.Numa)
	return nil
}

func DInit() error {
	if !is_init.Swap(false) {
		return errors.New("is dinit")
	}
	backend_dinit()
	return nil
}

func LIB_ggml_padding(u1, u2 uint64) uint64 {
	u2 -= 1
	return (u1 + u2) & (SIZE_MAX ^ u2)
}

func LIB_ggml_version() string {
	return C.GoString(C.ggml_version())
}

func LIB_ggml_commit() string {
	return C.GoString(C.ggml_commit())
}

// use this to compute the memory overhead of a tensor
func LIB_ggml_tensor_overhead() uint64 {
	// GGML_API size_t ggml_tensor_overhead(void);
	return uint64(C.ggml_tensor_overhead())
}

func LIB_ggml_graph_overhead_custom(n uint64) uint64 {
	// GGML_API size_t ggml_graph_overhead_custom(size_t size, bool grads);
	return uint64(C.ggml_graph_overhead_custom(C.size_t(n), false))
}

func LIB_ggml_blck_size(t ggmlgo.GGML_TYPE) int64 {
	// GGML_API int64_t ggml_blck_size(enum ggml_type type);
	return int64(C.ggml_blck_size(C.enum_ggml_type(t)))
}

func LIB_ggml_type_size(t ggmlgo.GGML_TYPE) uint64 {
	// GGML_API size_t  ggml_type_size(enum ggml_type type);             // size in bytes for all elements in a block
	return uint64(C.ggml_type_size(C.enum_ggml_type(t)))
}

func LIB_ggml_row_size(t ggmlgo.GGML_TYPE, ne int64) uint64 {
	// GGML_API size_t  ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row
	return uint64(C.ggml_row_size(C.enum_ggml_type(t), C.int64_t(ne)))
}

func LIB_ggml_ftype_to_ggml_type(t ggmlgo.GGML_FTYPE) ggmlgo.GGML_TYPE {
	// TODO: temporary until model loading of ggml examples is refactored
	// GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
	return ggmlgo.GGML_TYPE(C.ggml_ftype_to_ggml_type(C.enum_ggml_ftype(t)))
}

func LIB_ggml_is_quantized(t ggmlgo.GGML_TYPE) bool {
	// GGML_API bool    ggml_is_quantized(enum ggml_type type);
	return bool(C.ggml_is_quantized(C.enum_ggml_type(t)))
}

func LIB_ggml_type_name(t ggmlgo.GGML_TYPE) string {
	// GGML_API const char * ggml_type_name(enum ggml_type type);
	return C.GoString(C.ggml_type_name(C.enum_ggml_type(t)))
}

func LIB_ggml_op_name(t ggmlgo.GGML_OP) string {
	// GGML_API const char * ggml_op_name  (enum ggml_op   op);
	return C.GoString(C.ggml_op_name(C.enum_ggml_op(t)))
}

func LIB_ggml_op_symbol(t ggmlgo.GGML_OP) string {
	// GGML_API const char * ggml_op_symbol(enum ggml_op   op);
	return C.GoString(C.ggml_op_symbol(C.enum_ggml_op(t)))
}

func LIB_ggml_unary_op_name(t ggmlgo.GGML_UNARY_OP) string {
	// GGML_API const char * ggml_unary_op_name(enum ggml_unary_op op);
	return C.GoString(C.ggml_unary_op_name(C.enum_ggml_unary_op(t)))
}

func LIB_ggml_glu_op_name(t ggmlgo.GGML_GLU_OP) string {
	// GGML_API const char * ggml_glu_op_name(enum ggml_glu_op op);
	return C.GoString(C.ggml_glu_op_name(C.enum_ggml_glu_op(t)))
}

// ------------------------------------backend

func LIB_dev_count() uint64 {
	// GGML_API size_t             ggml_backend_dev_count(void);
	return uint64(C.ggml_backend_dev_count())
}

func LIB_reg_count() uint64 {
	// GGML_API size_t             ggml_backend_reg_count(void);
	return uint64(C.ggml_backend_reg_count())
}

func GetDevs() []Dev {
	var l []Dev
	for k, v := range devs {
		l = append(l, Dev{org: v, idx: uint8(k), Info: v.info()})
	}
	return l
}
