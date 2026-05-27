// @@
// @ Author       : Eacher
// @ Date         : 2026-05-25 11:35:55
// @ LastEditTime : 2026-05-27 15:18:30
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
	org  *ggml
	idx  int
	Name string
	T    ggmlgo.GGML_TYPE
	UOP  ggmlgo.GGML_UNARY_OP
	GOP  ggmlgo.GGML_GLU_OP
	OP   ggmlgo.GGML_OP // compute data
}

func (obj *Tensor) check_ggml() (*C.struct_ggml_tensor, error) {
	err := errors.New("not init")
	var tensor *C.struct_ggml_tensor
	if obj.org == nil {
		return tensor, err
	}
	if err = errors.New("ggml is close"); obj.org.is_close {
		return tensor, err
	}
	if err = errors.New("idx overris"); obj.idx < len(obj.org._tensors) {
		err, tensor = nil, obj.org._tensors[obj.idx]
	}
	return tensor, err
}

func (obj Tensor) P_nelements() (int64, error) {
	n := int64(0)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API int64_t ggml_nelements (const struct ggml_tensor * tensor);
		n = int64(C.ggml_nelements(_tensor))
	}
	return n, err
}

func (obj Tensor) P_gggml_nrows() (int64, error) {
	n := int64(0)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API int64_t ggml_nrows     (const struct ggml_tensor * tensor);
		n = int64(C.ggml_nrows(_tensor))
	}
	return n, err
}

func (obj Tensor) P_nbytes() (uint64, error) {
	n := uint64(0)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API size_t  ggml_nbytes    (const struct ggml_tensor * tensor);
		n = uint64(C.ggml_nrows(_tensor))
	}
	return n, err
}

func (obj Tensor) P_nbytes_pad() (uint64, error) {
	n := uint64(0)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API size_t  ggml_nbytes_pad(const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN
		n = uint64(C.ggml_nbytes_pad(_tensor))
	}
	return n, err
}

func (obj Tensor) P_op_desc() (string, error) {
	n := string("")
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API const char * ggml_op_desc(const struct ggml_tensor * t); // unary or op name
		n = C.GoString(C.ggml_op_desc(_tensor))
	}
	return n, err
}

func (obj Tensor) P_element_size() (uint64, error) {
	n := uint64(0)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);
		n = uint64(C.ggml_element_size(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_transposed() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_transposed(const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_transposed(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_permuted() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_permuted  (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_permuted(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_empty() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_empty     (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_empty(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_view() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_view      (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_view(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_scalar() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_scalar    (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_scalar(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_vector() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_vector    (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_vector(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_matrix() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_matrix    (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_matrix(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_3d() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_3d        (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_3d(_tensor))
	}
	return n, err
}

func (obj Tensor) P_n_dims() (int, error) {
	n := int(1)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars
		n = int(C.ggml_n_dims(_tensor))
	}
	return n, err
}

// returns whether the tensor elements can be iterated over with a flattened index (no gaps, no permutation)
func (obj Tensor) P_is_contiguous() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_contiguous  (const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_contiguous(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_contiguous_0() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_contiguous_0(const struct ggml_tensor * tensor); // same as ggml_is_contiguous()
		n = bool(C.ggml_is_contiguous_0(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_contiguous_1() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_contiguous_1(const struct ggml_tensor * tensor); // contiguous for dims >= 1
		n = bool(C.ggml_is_contiguous_1(_tensor))
	}
	return n, err
}

func (obj Tensor) P_is_contiguous_2() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_contiguous_2(const struct ggml_tensor * tensor); // contiguous for dims >= 2
		n = bool(C.ggml_is_contiguous_2(_tensor))
	}
	return n, err
}

// returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
func (obj Tensor) P_is_contiguously_allocated() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_contiguously_allocated(_tensor))
	}
	return n, err
}

// true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
func (obj Tensor) P_is_contiguous_channels() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_contiguous_channels(const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_contiguous_channels(_tensor))
	}
	return n, err
}

// true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
func (obj Tensor) P_is_contiguous_rows() (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err == nil {
		// GGML_API bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor);
		n = bool(C.ggml_is_contiguous_rows(_tensor))
	}
	return n, err
}

func (obj Tensor) P_are_same_shape(obj1 Tensor) (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err != nil {
		return n, err
	}
	_tensor1, err1 := obj1.check_ggml()
	if err1 != nil {
		return n, err1
	}
	// GGML_API bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	n = bool(C.ggml_are_same_shape(_tensor, _tensor1))
	return n, nil
}

func (obj Tensor) P_are_same_stride(obj1 Tensor) (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err != nil {
		return n, err
	}
	_tensor1, err1 := obj1.check_ggml()
	if err1 != nil {
		return n, err1
	}
	// GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	n = bool(C.ggml_are_same_stride(_tensor, _tensor1))
	return n, nil
}

func (obj Tensor) P_can_repeat(obj1 Tensor) (bool, error) {
	n := bool(false)
	_tensor, err := obj.check_ggml()
	if err != nil {
		return n, err
	}
	_tensor1, err1 := obj1.check_ggml()
	if err1 != nil {
		return n, err1
	}
	// GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	n = bool(C.ggml_can_repeat(_tensor, _tensor1))
	return n, nil
}

// -------------------------

// 复制
func (obj Tensor) Dup(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_DUP
	}
	return obj1, err
}

func (obj Tensor) SQR(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_SQR
	}
	return obj1, err
}

func (obj Tensor) SQRT(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_SQRT
	}
	return obj1, err
}

func (obj Tensor) LOG(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_LOG
	}
	return obj1, err
}

func (obj Tensor) SIN(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_SIN
	}
	return obj1, err
}

func (obj Tensor) COS(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_COS
	}
	return obj1, err
}

func (obj Tensor) Add(src Tensor, view bool) (Tensor, error) {
	is, err := src.P_can_repeat(obj)
	if err != nil {
		return Tensor{}, err
	}
	if !is {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	_tensor1, obj1 := src.org._tensors[src.idx], Tensor{}
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, _tensor1)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx, _tensor1)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_ADD
	}
	return obj1, err
}

func (obj Tensor) Add1(src Tensor, view bool) (Tensor, error) {
	if is, err := src.P_is_scalar(); err != nil {
		return Tensor{}, err
	} else if !is {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	// tensor->nb[0] == ggml_type_size(tensor->type) &&
	// tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
	// tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
	if _tensor0.nb[0] != C.ggml_type_size(_tensor0._type) ||
		_tensor0.nb[2] != (_tensor0.nb[1]*C.size_t(_tensor0.ne[1])) ||
		_tensor0.nb[3] != (_tensor0.nb[2]*C.size_t(_tensor0.ne[2])) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	_tensor1, obj1 := src.org._tensors[src.idx], Tensor{}
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, _tensor1)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx, _tensor1)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_ADD1
	}
	return obj1, err
}

func (obj Tensor) Add_ID(src Tensor, id Tensor) (Tensor, error) {
	if id.T != ggmlgo.GGML_TYPE_I32 {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := id.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, _tensor1, _tensor2 := obj.org._tensors[obj.idx], src.org._tensors[src.idx], id.org._tensors[id.idx]
	if _tensor0.ne[0] != _tensor1.ne[0] {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if _tensor0.ne[1] != _tensor2.ne[0] {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if _tensor0.ne[2] != _tensor2.ne[1] {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	obj1, err := obj.org.ggml_dup_tensor(obj.idx, _tensor1, _tensor2)
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_ADD_ID
	}
	return obj1, err
}

func (obj Tensor) ACC(src Tensor, view bool, params [4]int32) (Tensor, error) {
	if obj.T != ggmlgo.GGML_TYPE_F32 || src.T != ggmlgo.GGML_TYPE_F32 {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if _, err := obj.P_is_contiguous(); err != nil {
		return Tensor{}, err
	}
	n1, err := obj.P_nelements()
	if err != nil {
		return Tensor{}, err
	}
	n2, err := src.P_nelements()
	if err != nil {
		return Tensor{}, err
	}
	if n2 > n1 {
		return Tensor{}, errors.New("n2 > n1")
	}
	_tensor1 := src.org._tensors[src.idx]
	obj1, err := Tensor{}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, _tensor1)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx, _tensor1)
	}
	if err == nil {
		obj1.org._tensors[obj1.idx].op_params[0] = C.int32_t(params[0])
		obj1.org._tensors[obj1.idx].op_params[1] = C.int32_t(params[1])
		obj1.org._tensors[obj1.idx].op_params[2] = C.int32_t(params[2])
		obj1.org._tensors[obj1.idx].op_params[3] = C.int32_t(params[3])
		obj1.org._tensors[obj1.idx].op_params[4] = 0
		if view {
			obj1.org._tensors[obj1.idx].op_params[4] = 1
		}
		obj1.OP = ggmlgo.GGML_OP_ACC
	}
	return obj1, err
}

func (obj Tensor) SUB(src Tensor, view bool) (Tensor, error) {
	is, err := src.P_can_repeat(obj)
	if err != nil {
		return Tensor{}, err
	}
	if !is {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	_tensor1, obj1 := src.org._tensors[src.idx], Tensor{}
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, _tensor1)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx, _tensor1)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_SUB
	}
	return obj1, err
}

func (obj Tensor) MUL(src Tensor, view bool) (Tensor, error) {
	is, err := src.P_can_repeat(obj)
	if err != nil {
		return Tensor{}, err
	}
	if !is {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	_tensor1, obj1 := src.org._tensors[src.idx], Tensor{}
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, _tensor1)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx, _tensor1)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_MUL
	}
	return obj1, err
}

func (obj Tensor) DIV(src Tensor, view bool) (Tensor, error) {
	is, err := src.P_can_repeat(obj)
	if err != nil {
		return Tensor{}, err
	}
	if !is {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	_tensor1, obj1 := src.org._tensors[src.idx], Tensor{}
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, _tensor1)
	} else {
		obj1, err = obj.org.ggml_dup_tensor(obj.idx, _tensor1)
	}
	if err == nil {
		obj1.OP = ggmlgo.GGML_OP_DIV
	}
	return obj1, err
}
