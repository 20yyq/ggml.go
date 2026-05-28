// @@
// @ Author       : Eacher
// @ Date         : 2026-05-25 11:35:55
// @ LastEditTime : 2026-05-28 16:53:58
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
	view bool
	name string
	ne   [GGML_MAX_DIMS]int64  // number of elements
	nb   [GGML_MAX_DIMS]uint64 // stride in bytes:
	t    ggmlgo.GGML_TYPE
	op   ggmlgo.GGML_OP // compute data

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

func (obj *Tensor) ggml_nelements() int64 {
	n := max(int64(obj.ne[0]*obj.ne[1]*obj.ne[2]*obj.ne[3]), 0)
	return n
}

func (obj *Tensor) ggml_nrows() int64 {
	n := max(int64(obj.ne[1]*obj.ne[2]*obj.ne[3]), 0)
	return n
}

func (obj *Tensor) ggml_nbytes() uint64 {
	i, nbytes, blck_size := 0, LIB_ggml_type_size(obj.t), uint64(LIB_ggml_blck_size(obj.t))
	if obj.ne[i] > 0 && blck_size != 1 {
		i, nbytes = 1, uint64(obj.ne[0])*obj.nb[0]/blck_size
	}
	for ; i < GGML_MAX_DIMS; i++ {
		if obj.ne[i] <= 0 {
			return 0
		}
		nbytes += uint64((obj.ne[i] - 1)) * obj.nb[i]
	}
	return nbytes
}

func (obj Tensor) Name() string {
	return obj.name
}

func (obj Tensor) T() ggmlgo.GGML_TYPE {
	return obj.t
}

// func (obj Tensor) UOP() ggmlgo.GGML_UNARY_OP {
// 	return obj.t
// }

// func (obj Tensor) GOP() ggmlgo.GGML_GLU_OP {
// 	return obj.t
// }

func (obj Tensor) OP() ggmlgo.GGML_OP {
	return obj.op
}

func (obj Tensor) P_nelements() int64 {
	// GGML_API int64_t ggml_nelements (const struct ggml_tensor * tensor);
	return obj.ggml_nelements()
}

func (obj Tensor) P_ggml_nrows() int64 {
	// GGML_API int64_t ggml_nrows     (const struct ggml_tensor * tensor);
	return obj.ggml_nrows()
}

func (obj Tensor) P_nbytes() uint64 {
	// GGML_API size_t  ggml_nbytes    (const struct ggml_tensor * tensor);
	return obj.ggml_nbytes()
}

func (obj Tensor) P_nbytes_pad() uint64 {
	// GGML_API size_t  ggml_nbytes_pad(const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN
	return ggml_padding(obj.ggml_nbytes(), 16)
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

func (obj Tensor) P_element_size() uint64 {
	// GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);
	return LIB_ggml_type_size(obj.t)
}

func (obj Tensor) P_is_transposed() bool {
	// GGML_API bool ggml_is_transposed(const struct ggml_tensor * tensor);
	return obj.nb[0] > obj.nb[1]
}

func (obj Tensor) P_is_permuted() bool {
	// GGML_API bool ggml_is_permuted  (const struct ggml_tensor * tensor);
	// return tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
	return obj.nb[0] > obj.nb[1] || obj.nb[1] > obj.nb[2] || obj.nb[2] > obj.nb[3]
}

func (obj Tensor) P_is_empty() bool {
	// GGML_API bool ggml_is_empty     (const struct ggml_tensor * tensor);
	return obj.ne[0] == 0 || obj.ne[1] == 0 || obj.ne[2] == 0 || obj.ne[3] == 0
}

func (obj Tensor) P_is_view() bool {
	// GGML_API bool ggml_is_view      (const struct ggml_tensor * tensor);
	return obj.view
}

func (obj Tensor) P_is_scalar() bool {
	// GGML_API bool ggml_is_scalar    (const struct ggml_tensor * tensor);
	//  return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
	return obj.ne[0] == 1 && obj.ne[1] == 1 && obj.ne[2] == 1 && obj.ne[3] == 1
}

func (obj Tensor) P_is_vector() bool {
	// GGML_API bool ggml_is_vector    (const struct ggml_tensor * tensor);
	// return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
	return obj.ne[1] == 1 && obj.ne[2] == 1 && obj.ne[3] == 1
}

func (obj Tensor) P_is_matrix() bool {
	// GGML_API bool ggml_is_matrix    (const struct ggml_tensor * tensor);
	// tensor->ne[2] == 1 && tensor->ne[3] == 1;
	return obj.ne[2] == 1 && obj.ne[3] == 1
}

func (obj Tensor) P_is_3d() bool {
	// GGML_API bool ggml_is_3d        (const struct ggml_tensor * tensor);
	// obj.ne[3] == 1
	return obj.ne[3] == 1
}

func (obj Tensor) P_n_dims() int {
	// GGML_API int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars
	switch true {
	case obj.ne[3] > 1:
		return 4
	case obj.ne[2] > 1:
		return 3
	case obj.ne[1] > 1:
		return 2
	}
	return 1
}
func (obj *Tensor) is_contiguous(idx int) bool {
	next_nb, blck_size := LIB_ggml_type_size(obj.t), LIB_ggml_blck_size(obj.t)
	if obj.ne[0] != blck_size && obj.nb[0] != next_nb {
		return false
	}
	next_nb *= uint64(obj.ne[0] / blck_size)
	for i := 1; i < GGML_MAX_DIMS; i++ {
		if i > idx {
			if obj.ne[i] != 1 && obj.nb[i] != next_nb {
				return false
			}
			next_nb *= uint64(obj.ne[i])
		} else {
			// this dimension does not need to be contiguous
			next_nb = uint64(obj.ne[i]) * obj.nb[i]
		}
	}
	return true
}

// returns whether the tensor elements can be iterated over with a flattened index (no gaps, no permutation)
func (obj Tensor) P_is_contiguous() bool {
	// GGML_API bool ggml_is_contiguous  (const struct ggml_tensor * tensor);
	return obj.P_is_contiguous_0()
}

func (obj Tensor) P_is_contiguous_0() bool {
	// GGML_API bool ggml_is_contiguous_0(const struct ggml_tensor * tensor); // same as ggml_is_contiguous()
	return obj.is_contiguous(0)
}

func (obj Tensor) P_is_contiguous_1() bool {
	// GGML_API bool ggml_is_contiguous_1(const struct ggml_tensor * tensor); // contiguous for dims >= 1
	return obj.is_contiguous(1)
}

func (obj Tensor) P_is_contiguous_2() bool {
	// GGML_API bool ggml_is_contiguous_2(const struct ggml_tensor * tensor); // contiguous for dims >= 2
	return obj.is_contiguous(2)
}

// returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
func (obj Tensor) P_is_contiguously_allocated() bool {
	// GGML_API bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor);
	// return ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
	return obj.ggml_nbytes() == uint64(obj.ggml_nelements())*LIB_ggml_type_size(obj.t)/uint64(LIB_ggml_blck_size(obj.t))
}

// true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
func (obj Tensor) P_is_contiguous_channels() bool {
	// GGML_API bool ggml_is_contiguous_channels(const struct ggml_tensor * tensor);
	// return
	// tensor->nb[0] > tensor->nb[2] &&
	// tensor->nb[1] > tensor->nb[0] &&
	// tensor->nb[2] == ggml_type_size(tensor->type);
	return obj.nb[0] > obj.nb[2] && obj.nb[1] > obj.nb[0] && obj.nb[2] == LIB_ggml_type_size(obj.t)
}

// true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
func (obj Tensor) P_is_contiguous_rows() bool {
	// GGML_API bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor);
	// tensor->ne[0] == ggml_blck_size(tensor->type) ||
	// tensor->nb[0] == ggml_type_size(tensor->type);
	return obj.ne[0] == LIB_ggml_blck_size(obj.t) || obj.nb[0] == LIB_ggml_type_size(obj.t)
}

func (obj Tensor) P_are_same_shape(obj1 Tensor) bool {
	// GGML_API bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	// (t0->ne[0] == t1->ne[0]) &&
	// (t0->ne[1] == t1->ne[1]) &&
	// (t0->ne[2] == t1->ne[2]) &&
	// (t0->ne[3] == t1->ne[3]);
	return obj.ne[0] == obj1.ne[0] && obj.ne[1] == obj1.ne[1] && obj.ne[2] == obj1.ne[2] && obj.ne[3] == obj1.ne[3]
}

func (obj Tensor) P_are_same_stride(obj1 Tensor) bool {
	// GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	// (t0->nb[0] == t1->nb[0]) &&
	// (t0->nb[1] == t1->nb[1]) &&
	// (t0->nb[2] == t1->nb[2]) &&
	// (t0->nb[3] == t1->nb[3]);
	return obj.nb[0] == obj1.nb[0] && obj.nb[1] == obj1.nb[1] && obj.nb[2] == obj1.nb[2] && obj.nb[3] == obj1.nb[3]
}

func (obj Tensor) P_can_repeat(obj1 Tensor) bool {
	// GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	// ggml_is_empty(t0) ? ggml_is_empty(t1) :
	//     (t1->ne[0]%t0->ne[0] == 0) &&
	//     (t1->ne[1]%t0->ne[1] == 0) &&
	//     (t1->ne[2]%t0->ne[2] == 0) &&
	//     (t1->ne[3]%t0->ne[3] == 0);
	if obj.P_is_empty() {
		return obj1.P_is_empty()
	}
	return (obj1.ne[0]%obj.ne[0] == 0) && (obj1.ne[1]%obj.ne[1] == 0) && (obj1.ne[2]%obj.ne[2] == 0) && (obj1.ne[3]%obj.ne[3] == 0)
}

// -------------------------

// 复制
func (obj Tensor) Dup(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{op: ggmlgo.GGML_OP_DUP}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) SQR(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{op: ggmlgo.GGML_OP_SQR}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) SQRT(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{op: ggmlgo.GGML_OP_SQRT}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) LOG(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{op: ggmlgo.GGML_OP_LOG}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) SIN(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{op: ggmlgo.GGML_OP_SIN}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) COS(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{op: ggmlgo.GGML_OP_COS}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) Add(src Tensor, view bool) (Tensor, error) {
	if !src.P_can_repeat(obj) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1, obj1, err := src.org._tensors[src.idx], Tensor{op: ggmlgo.GGML_OP_ADD}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op, _tensor1)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op, _tensor1)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) Add1(src Tensor, view bool) (Tensor, error) {
	if !src.P_is_scalar() {
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
	if _, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1, obj1 := src.org._tensors[src.idx], Tensor{op: ggmlgo.GGML_OP_ADD1}
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op, _tensor1)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op, _tensor1)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) Add_ID(src Tensor, id Tensor) (Tensor, error) {
	if id.t != ggmlgo.GGML_TYPE_I32 {
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
	obj1, err := obj.org.ggml_new_tensor(obj.t, obj.ne[:], ggmlgo.GGML_OP_ADD_ID, _tensor1, _tensor2)
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) ACC(src Tensor, view bool, params [4]int32) (Tensor, error) {
	if obj.t != ggmlgo.GGML_TYPE_F32 || src.t != ggmlgo.GGML_TYPE_F32 {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if !obj.P_is_contiguous() {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	n1, n2 := obj.ggml_nelements(), src.ggml_nelements()
	if n2 > n1 {
		return Tensor{}, errors.New("n2 > n1")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1 := src.org._tensors[src.idx]
	obj1, err := Tensor{op: ggmlgo.GGML_OP_ACC}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op, _tensor1)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op, _tensor1)
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
		obj1.op = ggmlgo.GGML_OP_ACC
	}
	return obj1, err
}

func (obj Tensor) SUB(src Tensor, view bool) (Tensor, error) {
	if !src.P_can_repeat(obj) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1, obj1, err := src.org._tensors[src.idx], Tensor{op: ggmlgo.GGML_OP_SUB}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op, _tensor1)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op, _tensor1)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) MUL(src Tensor, view bool) (Tensor, error) {
	if !src.P_can_repeat(obj) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1, obj1, err := src.org._tensors[src.idx], Tensor{op: ggmlgo.GGML_OP_MUL}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op, _tensor1)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op, _tensor1)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) DIV(src Tensor, view bool) (Tensor, error) {
	if !src.P_can_repeat(obj) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1, obj1, err := src.org._tensors[src.idx], Tensor{op: ggmlgo.GGML_OP_DIV}, error(nil)
	if view {
		obj1, err = obj.org.ggml_view_tensor(obj.idx, obj1.op, _tensor1)
	} else {
		obj1, err = obj.org.ggml_new_tensor(obj.t, obj.ne[:], obj1.op, _tensor1)
	}
	if err != nil {
		obj1.op = ggmlgo.GGML_OP_COUNT
	}
	return obj1, err
}

func (obj Tensor) SUM() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, b := obj.org._tensors[obj.idx], []int64{1}
	obj1, err := obj.org.ggml_new_tensor(obj.t, b, ggmlgo.GGML_OP_SUM, _tensor0)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) SUM_ROWS() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := obj.org.ggml_new_tensor(obj.t, obj.ne[:], ggmlgo.GGML_OP_SUM_ROWS)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) CUMSUM() (Tensor, error) {
	if obj.t != ggmlgo.GGML_TYPE_F32 {
		return Tensor{}, errors.New("obj.t != GGML_TYPE_F32")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := obj.org.ggml_new_tensor(obj.t, obj.ne[:], ggmlgo.GGML_OP_CUMSUM)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) MEAN() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, b := obj.org._tensors[obj.idx], obj.ne[:]
	b[0] = 1
	obj1, err := obj.org.ggml_new_tensor(ggmlgo.GGML_TYPE_F32, b, ggmlgo.GGML_OP_MEAN, _tensor0)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) ARGMAX() (Tensor, error) {
	if obj.ne[0] > 2147483647 {
		return Tensor{}, errors.New("> INT32_MAX")
	}
	if !obj.P_is_matrix() {
		return Tensor{}, errors.New("!obj.P_is_matrix()")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, b := obj.org._tensors[obj.idx], []int64{obj.ne[1]}
	obj1, err := obj.org.ggml_new_tensor(ggmlgo.GGML_TYPE_I32, b, ggmlgo.GGML_OP_ARGMAX, _tensor0)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) COUNT_EQUAL(src Tensor) (Tensor, error) {
	if !obj.P_are_same_shape(src) {
		return Tensor{}, errors.New("!obj.P_are_same_shape()")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, _tensor1, b := obj.org._tensors[obj.idx], obj.org._tensors[src.idx], []int64{1}
	obj1, err := obj.org.ggml_new_tensor(ggmlgo.GGML_TYPE_I64, b, ggmlgo.GGML_OP_COUNT_EQUAL, _tensor0, _tensor1)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) REPEAT(src Tensor) (Tensor, error) {
	obj1, err := obj.REPEAT_4D(src.ne)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) REPEAT_4D(b1 [4]int64) (Tensor, error) {
	if !obj.P_is_empty() && !obj.P_can_repeat(Tensor{ne: b1}) {
		return Tensor{}, errors.New("!obj.P_is_empty() && !obj.P_can_repeat(src)")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, b := obj.org._tensors[obj.idx], b1[:]
	obj1, err := obj.org.ggml_new_tensor(obj.t, b, ggmlgo.GGML_OP_REPEAT, _tensor0)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) REPEAT_BACK(src Tensor) (Tensor, error) {
	if !obj.P_can_repeat(src) {
		return Tensor{}, errors.New("!obj.P_can_repeat()")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, b := obj.org._tensors[obj.idx], src.ne[:]
	obj1, err := obj.org.ggml_new_tensor(obj.t, b, ggmlgo.GGML_OP_REPEAT_BACK, _tensor0)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}
