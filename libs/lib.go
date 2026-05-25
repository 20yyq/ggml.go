// @@
// @ Author       : Eacher
// @ Date         : 2026-05-22 08:21:47
// @ LastEditTime : 2026-05-23 22:06:54
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

// #cgo CFLAGS: -I${SRCDIR}/../ggml/include
// #cgo CPPFLAGS: -I${SRCDIR}/../ggml/include
// #cgo LDFLAGS: -lstdc++ -lpthread
// #cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml -lggml-base -lggml-cuda -lggml-cpu-x64
// #cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml-cpu-alderlake -lggml-cpu-cannonlake -lggml-cpu-cascadelake
// #cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml-cpu-cooperlake -lggml-cpu-haswell -lggml-cpu-icelake
// #cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml-cpu-ivybridge -lggml-cpu-piledriver -lggml-cpu-sandybridge
// #cgo LDFLAGS: -L${SRCDIR}/build/bin -lggml-cpu-sapphirerapids -lggml-cpu-skylakex -lggml-cpu-sse42 -lggml-cpu-zen4
//
// #include "ggml.h"
// #include "ggml-backend.h"
// #include "gguf.h"
//
// extern void go_log_callback(enum ggml_log_level level, char * text, void * user_data);
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

// #cgo CFLAGS: -I${SRCDIR}/ggml/include
// #cgo CPPFLAGS: -I${SRCDIR}/ggml/include
// #cgo LDFLAGS: -L${SRCDIR}/build/ggml/src -lstdc++ -lm -lggml -lggml-base -lggml-cpu
// #cgo LDFLAGS: -L${SRCDIR}/build/ggml/src/ggml-cuda -lstdc++ -lm -lggml-cuda

//export go_log_callback
func go_log_callback(level C.enum_ggml_log_level, text *C.char, _ unsafe.Pointer) {
	fmt.Printf("%d %s", level, C.GoString(text))
}

func init() {
	C.ggml_time_init()
	C.ggml_log_set(C.ggml_log_callback(C.go_log_callback), nil)

	// needed to initialize f16 tables
	{
		var params C.struct_ggml_init_params
		var ctx *C.struct_ggml_context = C.ggml_init(params)
		C.ggml_free(ctx)
		fmt.Printf("%p \n", ctx)
	}

	if C.ggml_backend_load_all(); C.ggml_backend_reg_count() == 0 {
		// hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function
		panic("no backends are loaded.")
	}

}

type ggml struct {
	_gctx *C.struct_ggml_context
	_fctx *C.struct_gguf_context
}

func (gl *GGML) init(m Model) error {
	err := errors.New("not gguf")
	if gl.org._fctx == nil {
		return err
	}
	defer func() {
		C.gguf_free(gl.org._fctx)
		gl.org._fctx = nil
	}()
	if err = errors.New("not ggml"); gl.org._gctx == nil {
		return err
	}
	defer func() {
		C.ggml_free(gl.org._gctx)
		gl.org._gctx = nil
	}()
	m.Loader(gl.org)
	m.Devices(gl.org)
	m.LoadHparams(gl.org)
	m.LoadVocab(gl.org)
	m.LoadStatus(gl.org)
	m.LoadTensors(gl.org)
	return nil
}

// var c1 C.enum_ggml_backend_dev_type = C.GGML_BACKEND_DEVICE_TYPE_CPU
// fmt.Printf("%p %p \n", C.ggml_backend_init_best(), C.ggml_backend_dev_by_type(c1))
func Test() {
	// C.ggml_quantize_free()
	// C.gguf_init_empty()
}

func (org ggml) ggml_blck_size(GGML_IFACE) int64 {
	// GGML_API int64_t ggml_blck_size(enum ggml_type type);
	return 0
}

func (org ggml) ggml_type_size(GGML_IFACE) uint64 {
	// GGML_API size_t  ggml_type_size(enum ggml_type type);             // size in bytes for all elements in a block
	return 0
}

func (org ggml) ggml_row_size(GGML_IFACE) uint64 {
	// GGML_API size_t  ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row
	return 0
}

func (org ggml) ggml_nelements(GGML_IFACE) int64 {
	// GGML_API int64_t ggml_nelements (const struct ggml_tensor * tensor);
	return 0
}

func (org ggml) gggml_nrows(GGML_IFACE) int64 {
	// GGML_API int64_t ggml_nrows     (const struct ggml_tensor * tensor);
	return 0
}

func (org ggml) ggml_nbytes(GGML_IFACE) uint64 {
	// GGML_API size_t  ggml_nbytes    (const struct ggml_tensor * tensor);
	return 0
}

func (org ggml) ggml_nbytes_pad(GGML_IFACE) uint64 {
	// GGML_API size_t  ggml_nbytes_pad(const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN
	return 0
}

func (org ggml) ggml_op_desc(GGML_IFACE) string {
	// GGML_API const char * ggml_op_desc(const struct ggml_tensor * t); // unary or op name
	return C.GoString(C.ggml_op_desc(nil))
}

func (org ggml) ggml_element_size(GGML_IFACE) uint64 {
	// GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);
	return 0
}

func (org ggml) ggml_is_transposed(GGML_IFACE) bool {
	// GGML_API bool ggml_is_transposed(const struct ggml_tensor * tensor);
	return false
}

func (org ggml) ggml_is_permuted(GGML_IFACE) bool {
	// GGML_API bool ggml_is_permuted  (const struct ggml_tensor * tensor);
	return false
}

func (org ggml) ggml_is_empty(GGML_IFACE) bool {
	// GGML_API bool ggml_is_empty     (const struct ggml_tensor * tensor);
	return false
}

func (org ggml) ggml_is_view(GGML_IFACE) bool {
	// GGML_API bool ggml_is_view      (const struct ggml_tensor * tensor);
	return false
}

func (org ggml) ggml_is_scalar(GGML_IFACE) bool {
	// GGML_API bool ggml_is_scalar    (const struct ggml_tensor * tensor);
	return false
}

func (org ggml) ggml_is_vector(GGML_IFACE) bool {
	// GGML_API bool ggml_is_vector    (const struct ggml_tensor * tensor);
	return false
}

func (org ggml) ggml_is_matrix(GGML_IFACE) bool {
	// GGML_API bool ggml_is_matrix    (const struct ggml_tensor * tensor);
	return false
}

func (org ggml) ggml_is_3d(GGML_IFACE) bool {
	// GGML_API bool ggml_is_3d        (const struct ggml_tensor * tensor);
	return false
}

func (org ggml) ggml_n_dims(GGML_IFACE) int {
	// GGML_API int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars
	return 1
}

// returns whether the tensor elements can be iterated over with a flattened index (no gaps, no permutation)
func (org ggml) ggml_is_contiguous(GGML_IFACE) bool {
	// GGML_API bool ggml_is_contiguous  (const struct ggml_tensor * tensor);
	return false
}

func (org ggml) ggml_is_contiguous_0(GGML_IFACE) bool {
	// GGML_API bool ggml_is_contiguous_0(const struct ggml_tensor * tensor); // same as ggml_is_contiguous()
	return false
}

func (org ggml) ggml_is_contiguous_1(GGML_IFACE) bool {
	// GGML_API bool ggml_is_contiguous_1(const struct ggml_tensor * tensor); // contiguous for dims >= 1
	return false
}

func (org ggml) ggml_is_contiguous_2(GGML_IFACE) bool {
	// GGML_API bool ggml_is_contiguous_2(const struct ggml_tensor * tensor); // contiguous for dims >= 2
	return false
}

// returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
func (org ggml) ggml_is_contiguously_allocated(GGML_IFACE) bool {
	// GGML_API bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor);
	return false
}

// true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
func (org ggml) ggml_is_contiguous_channels(GGML_IFACE) bool {
	// GGML_API bool ggml_is_contiguous_channels(const struct ggml_tensor * tensor);
	return false
}

// true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
func (org ggml) ggml_is_contiguous_rows(GGML_IFACE) bool {
	// GGML_API bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor);
	return false
}

func (org ggml) ggml_are_same_shape(GGML_IFACE) bool {
	// GGML_API bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	return false
}

func (org ggml) ggml_are_same_stride(GGML_IFACE) bool {
	// GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	return false
}

func (org ggml) ggml_can_repeat(GGML_IFACE) bool {
	// GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	return false
}
