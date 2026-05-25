// @@
// @ Author       : Eacher
// @ Date         : 2026-05-25 11:35:55
// @ LastEditTime : 2026-05-25 17:44:11
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

// #include "ggml.h"
import "C"
import (
	"errors"

	ggmlgo "ggml.go"
)

type Tensor struct {
	org     *ggml
	_tensor *C.struct_ggml_tensor
	Name    string
	T       ggmlgo.GGML_TYPE
	UOP     ggmlgo.GGML_UNARY_OP
	GOP     ggmlgo.GGML_GLU_OP
	OP      ggmlgo.GGML_OP // compute data
}

func (obj *Tensor) check_ggml() error {
	err := errors.New("not init")
	if obj._tensor != nil {
		if err = nil; obj.org.is_close {
			err = errors.New("ggml is close")
		}
	}
	return err
}

func (obj Tensor) P_nelements() (int64, error) {
	err, n := obj.check_ggml(), int64(0)
	if err == nil {
		// GGML_API int64_t ggml_nelements (const struct ggml_tensor * tensor);
		n = int64(C.ggml_nelements(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_gggml_nrows() (int64, error) {
	err, n := obj.check_ggml(), int64(0)
	if err == nil {
		// GGML_API int64_t ggml_nrows     (const struct ggml_tensor * tensor);
		n = int64(C.ggml_nrows(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_nbytes() (uint64, error) {
	err, n := obj.check_ggml(), uint64(0)
	if err == nil {
		// GGML_API size_t  ggml_nbytes    (const struct ggml_tensor * tensor);
		n = uint64(C.ggml_nrows(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_nbytes_pad() (uint64, error) {
	err, n := obj.check_ggml(), uint64(0)
	if err == nil {
		// GGML_API size_t  ggml_nbytes_pad(const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN
		n = uint64(C.ggml_nbytes_pad(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_op_desc() (string, error) {
	err, n := obj.check_ggml(), string("")
	if err == nil {
		// GGML_API const char * ggml_op_desc(const struct ggml_tensor * t); // unary or op name
		n = C.GoString(C.ggml_op_desc(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_element_size() (uint64, error) {
	err, n := obj.check_ggml(), uint64(0)
	if err == nil {
		// GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);
		n = uint64(C.ggml_element_size(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_transposed() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_transposed(const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_transposed(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_permuted() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_permuted  (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_permuted(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_empty() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_empty     (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_empty(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_view() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_view      (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_view(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_scalar() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_scalar    (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_scalar(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_vector() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_vector    (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_vector(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_matrix() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_matrix    (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_matrix(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_3d() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_3d        (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_3d(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_n_dims() (int, error) {
	err, n := obj.check_ggml(), int(1)
	if err == nil {
		// GGML_API int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars
		n = int(C.ggml_n_dims(obj._tensor))
	}
	return n, err
}

// returns whether the tensor elements can be iterated over with a flattened index (no gaps, no permutation)
func (obj Tensor) P_is_contiguous() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_contiguous  (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_contiguous(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_contiguous_0() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_contiguous_0(const struct ggml_tensor * tensor); // same as ggml_is_contiguous()
		n = bool(C.ggml_is_contiguous_0(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_contiguous_1() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_contiguous_1(const struct ggml_tensor * tensor); // contiguous for dims >= 1
		n = bool(C.ggml_is_contiguous_1(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_is_contiguous_2() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_contiguous_2(const struct ggml_tensor * tensor); // contiguous for dims >= 2
		n = bool(C.ggml_is_contiguous_2(obj._tensor))
	}
	return n, err
}

// returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
func (obj Tensor) P_is_contiguously_allocated() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_contiguously_allocated(obj._tensor))
	}
	return n, err
}

// true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
func (obj Tensor) P_is_contiguous_channels() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_contiguous_channels(const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_contiguous_channels(obj._tensor))
	}
	return n, err
}

// true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
func (obj Tensor) P_is_contiguous_rows() (bool, error) {
	err, n := obj.check_ggml(), bool(false)
	if err == nil {
		// GGML_API bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_contiguous_rows(obj._tensor))
	}
	return n, err
}

func (obj Tensor) P_are_same_shape(obj1 Tensor) (bool, error) {
	err, err1, n := obj.check_ggml(), obj1.check_ggml(), bool(false)
	if err != nil {
		return n, err
	}
	if err1 != nil {
		return n, err1
	}
	// GGML_API bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	n = bool(C.ggml_are_same_shape(obj._tensor, obj1._tensor))
	return n, nil
}

func (obj Tensor) P_are_same_stride(obj1 Tensor) (bool, error) {
	err, err1, n := obj.check_ggml(), obj1.check_ggml(), bool(false)
	if err != nil {
		return n, err
	}
	if err1 != nil {
		return n, err1
	}
	// GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	n = bool(C.ggml_are_same_stride(obj._tensor, obj1._tensor))
	return n, nil
}

func (obj Tensor) P_can_repeat(obj1 Tensor) (bool, error) {
	err, err1, n := obj.check_ggml(), obj1.check_ggml(), bool(false)
	if err != nil {
		return n, err
	}
	if err1 != nil {
		return n, err1
	}
	// GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	n = bool(C.ggml_can_repeat(obj._tensor, obj1._tensor))
	return n, nil
}
