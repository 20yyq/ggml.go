// @@
// @ Author       : Eacher
// @ Date         : 2026-05-25 11:35:55
// @ LastEditTime : 2026-06-29 14:13:38
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
	"unsafe"

	ggmlgo "ggml.go"
)

type TensorInfo struct {
	idx    int
	view   bool
	Name   string
	NE     [GGML_MAX_DIMS]int64  // number of elements
	NB     [GGML_MAX_DIMS]uint64 // stride in bytes:
	T      ggmlgo.GGML_TYPE
	OP     ggmlgo.GGML_OP // compute data
	Offset uint64
}

func (obj *TensorInfo) from_ggml_tensor(tensor *C.struct_ggml_tensor) {
	var ne []int64
	ne = unsafe.Slice((*int64)(unsafe.Pointer(&tensor.ne[0])), 4)
	copy(obj.NE[:], ne)
	var nb []uint64
	nb = unsafe.Slice((*uint64)(unsafe.Pointer(&tensor.nb[0])), 4)
	copy(obj.NB[:], nb)
	obj.T, obj.OP, obj.Name = ggmlgo.GGML_TYPE(tensor._type), ggmlgo.GGML_OP(tensor.op), C.GoString(&tensor.name[0])
}

func (obj TensorInfo) ggml_nelements() int64 {
	n := max(int64(obj.NE[0]*obj.NE[1]*obj.NE[2]*obj.NE[3]), 0)
	return n
}

func (obj TensorInfo) ggml_nrows() int64 {
	n := max(int64(obj.NE[1]*obj.NE[2]*obj.NE[3]), 0)
	return n
}

func (obj TensorInfo) ggml_nbytes() uint64 {
	i, nbytes, blck_size := 0, LIB_ggml_type_size(obj.T), uint64(LIB_ggml_blck_size(obj.T))
	if obj.NE[i] > 0 && blck_size != 1 {
		i, nbytes = 1, uint64(obj.NE[0])*obj.NB[0]/blck_size
	}
	for ; i < GGML_MAX_DIMS; i++ {
		if obj.NE[i] <= 0 {
			return 0
		}
		nbytes += uint64((obj.NE[i] - 1)) * obj.NB[i]
	}
	return nbytes
}

func (obj TensorInfo) P_nelements() int64 {
	// GGML_API int64_t ggml_nelements (const struct ggml_tensor * tensor);
	return obj.ggml_nelements()
}

func (obj TensorInfo) P_ggml_nrows() int64 {
	// GGML_API int64_t ggml_nrows     (const struct ggml_tensor * tensor);
	return obj.ggml_nrows()
}

func (obj TensorInfo) P_nbytes() uint64 {
	// GGML_API size_t  ggml_nbytes    (const struct ggml_tensor * tensor);
	return obj.ggml_nbytes()
}

func (obj TensorInfo) P_nbytes_pad() uint64 {
	// GGML_API size_t  ggml_nbytes_pad(const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN
	return LIB_ggml_padding(obj.ggml_nbytes(), 16)
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

func (obj TensorInfo) P_element_size() uint64 {
	// GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);
	return LIB_ggml_type_size(obj.T)
}

func (obj TensorInfo) P_is_transposed() bool {
	// GGML_API bool ggml_is_transposed(const struct ggml_tensor * tensor);
	return obj.NB[0] > obj.NB[1]
}

func (obj TensorInfo) P_is_permuted() bool {
	// GGML_API bool ggml_is_permuted  (const struct ggml_tensor * tensor);
	// return tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
	return obj.NB[0] > obj.NB[1] || obj.NB[1] > obj.NB[2] || obj.NB[2] > obj.NB[3]
}

func (obj TensorInfo) P_is_empty() bool {
	// GGML_API bool ggml_is_empty     (const struct ggml_tensor * tensor);
	return obj.NE[0] == 0 || obj.NE[1] == 0 || obj.NE[2] == 0 || obj.NE[3] == 0
}

func (obj TensorInfo) P_is_scalar() bool {
	// GGML_API bool ggml_is_scalar    (const struct ggml_tensor * tensor);
	//  return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
	return obj.NE[0] == 1 && obj.NE[1] == 1 && obj.NE[2] == 1 && obj.NE[3] == 1
}

func (obj TensorInfo) P_is_vector() bool {
	// GGML_API bool ggml_is_vector    (const struct ggml_tensor * tensor);
	// return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
	return obj.NE[1] == 1 && obj.NE[2] == 1 && obj.NE[3] == 1
}

func (obj TensorInfo) P_is_matrix() bool {
	// GGML_API bool ggml_is_matrix    (const struct ggml_tensor * tensor);
	// tensor->ne[2] == 1 && tensor->ne[3] == 1;
	return obj.NE[2] == 1 && obj.NE[3] == 1
}

func (obj TensorInfo) P_is_3d() bool {
	// GGML_API bool ggml_is_3d        (const struct ggml_tensor * tensor);
	// obj.NE[3] == 1
	return obj.NE[3] == 1
}

func (obj TensorInfo) P_n_dims() int {
	// GGML_API int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars
	switch true {
	case obj.NE[3] > 1:
		return 4
	case obj.NE[2] > 1:
		return 3
	case obj.NE[1] > 1:
		return 2
	}
	return 1
}
func (obj *TensorInfo) is_contiguous(idx int) bool {
	next_nb, blck_size := LIB_ggml_type_size(obj.T), LIB_ggml_blck_size(obj.T)
	if obj.NE[0] != blck_size && obj.NB[0] != next_nb {
		return false
	}
	next_nb *= uint64(obj.NE[0] / blck_size)
	for i := 1; i < GGML_MAX_DIMS; i++ {
		if i > idx {
			if obj.NE[i] != 1 && obj.NB[i] != next_nb {
				return false
			}
			next_nb *= uint64(obj.NE[i])
		} else {
			// this dimension does not need to be contiguous
			next_nb = uint64(obj.NE[i]) * obj.NB[i]
		}
	}
	return true
}

// returns whether the tensor elements can be iterated over with a flattened index (no gaps, no permutation)
func (obj TensorInfo) P_is_contiguous() bool {
	// GGML_API bool ggml_is_contiguous  (const struct ggml_tensor * tensor);
	return obj.P_is_contiguous_0()
}

func (obj TensorInfo) P_is_contiguous_0() bool {
	// GGML_API bool ggml_is_contiguous_0(const struct ggml_tensor * tensor); // same as ggml_is_contiguous()
	return obj.is_contiguous(0)
}

func (obj TensorInfo) P_is_contiguous_1() bool {
	// GGML_API bool ggml_is_contiguous_1(const struct ggml_tensor * tensor); // contiguous for dims >= 1
	return obj.is_contiguous(1)
}

func (obj TensorInfo) P_is_contiguous_2() bool {
	// GGML_API bool ggml_is_contiguous_2(const struct ggml_tensor * tensor); // contiguous for dims >= 2
	return obj.is_contiguous(2)
}

// returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
func (obj TensorInfo) P_is_contiguously_allocated() bool {
	// GGML_API bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor);
	// return ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type)/ggml_blck_size(tensor->type);
	return obj.ggml_nbytes() == uint64(obj.ggml_nelements())*LIB_ggml_type_size(obj.T)/uint64(LIB_ggml_blck_size(obj.T))
}

// true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
func (obj TensorInfo) P_is_contiguous_channels() bool {
	// GGML_API bool ggml_is_contiguous_channels(const struct ggml_tensor * tensor);
	// return
	// tensor->nb[0] > tensor->nb[2] &&
	// tensor->nb[1] > tensor->nb[0] &&
	// tensor->nb[2] == ggml_type_size(tensor->type);
	return obj.NB[0] > obj.NB[2] && obj.NB[1] > obj.NB[0] && obj.NB[2] == LIB_ggml_type_size(obj.T)
}

// true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
func (obj TensorInfo) P_is_contiguous_rows() bool {
	// GGML_API bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor);
	// tensor->ne[0] == ggml_blck_size(tensor->type) ||
	// tensor->nb[0] == ggml_type_size(tensor->type);
	return obj.NE[0] == LIB_ggml_blck_size(obj.T) || obj.NB[0] == LIB_ggml_type_size(obj.T)
}

func (obj TensorInfo) P_are_same_shape(obj1 TensorInfo) bool {
	// GGML_API bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	// (t0->ne[0] == t1->ne[0]) &&
	// (t0->ne[1] == t1->ne[1]) &&
	// (t0->ne[2] == t1->ne[2]) &&
	// (t0->ne[3] == t1->ne[3]);
	return obj.NE[0] == obj1.NE[0] && obj.NE[1] == obj1.NE[1] && obj.NE[2] == obj1.NE[2] && obj.NE[3] == obj1.NE[3]
}

func (obj TensorInfo) P_are_same_stride(obj1 TensorInfo) bool {
	// GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	// (t0->nb[0] == t1->nb[0]) &&
	// (t0->nb[1] == t1->nb[1]) &&
	// (t0->nb[2] == t1->nb[2]) &&
	// (t0->nb[3] == t1->nb[3]);
	return obj.NB[0] == obj1.NB[0] && obj.NB[1] == obj1.NB[1] && obj.NB[2] == obj1.NB[2] && obj.NB[3] == obj1.NB[3]
}

func (obj TensorInfo) P_can_repeat(obj1 TensorInfo) bool {
	// GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	// ggml_is_empty(t0) ? ggml_is_empty(t1) :
	//     (t1->ne[0]%t0->ne[0] == 0) &&
	//     (t1->ne[1]%t0->ne[1] == 0) &&
	//     (t1->ne[2]%t0->ne[2] == 0) &&
	//     (t1->ne[3]%t0->ne[3] == 0);
	if obj.P_is_empty() {
		return obj1.P_is_empty()
	}
	return (obj1.NE[0]%obj.NE[0] == 0) && (obj1.NE[1]%obj.NE[1] == 0) && (obj1.NE[2]%obj.NE[2] == 0) && (obj1.NE[3]%obj.NE[3] == 0)
}

func (obj TensorInfo) P_can_repeat_rows(obj1 TensorInfo) bool {
	// GGML_API bool ggml_can_repeat_rows(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
	//  return (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
	return obj.P_can_repeat(obj1) && (obj.NE[0] == obj1.NE[0])
}

func (obj TensorInfo) P_is_view() bool {
	// GGML_API bool ggml_is_view      (const struct ggml_tensor * tensor);
	return obj.view
}

func (obj TensorInfo) P_can_mul_mat(obj1 TensorInfo) bool {
	// static inline bool ggml_can_mul_mat(const struct ggml_tensor * t0, const struct ggml_tensor * t1)
	// return (t0->ne[0]           == t1->ne[0])  &&
	//        (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
	//        (t1->ne[3]%t0->ne[3] == 0);
	return obj.NE[0] == obj1.NE[0] && (obj1.NE[2]%obj.NE[2] == 0) && (obj1.NE[3]%obj.NE[3] == 0)
}

// -------------------------

type Tensor struct {
	org  *GGML
	info TensorInfo
}

func (obj *Tensor) Init(ptr *GGML, idx int, info *TensorInfo) error {
	err := errors.New("not init")
	if !ptr.is_init || ptr.is_close {
		return err
	}
	if len(ptr._tensors) != idx {
		return err
	}
	obj.info, err = ptr.ggml_new_tensor(info.T, info.NE[:], ggmlgo.GGML_OP_NONE)
	if err == nil {
		obj.org, obj.info.Name = ptr, info.Name
		if _tensor, _ := obj.check_ggml(); _tensor != nil {
			hp := (*byte)(unsafe.Pointer(&_tensor.name[0]))
			var name []byte
			name = unsafe.Slice(hp, 63)
			copy(name, info.Name)
		}
		*info = obj.info
	}
	return err
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
	if err = errors.New("idx overris"); obj.info.idx < len(obj.org._tensors) {
		err, tensor = nil, obj.org._tensors[obj.info.idx]
	}
	return tensor, err
}

func (obj Tensor) Info() TensorInfo {
	return obj.info
}

func (obj Tensor) SetData(data []byte) error {
	tensor, err := obj.check_ggml()
	if err != nil {
		return err
	}
	l := uint64(len(data))
	if l < 1 || l != obj.info.ggml_nbytes() {
		return errors.New("l < 1 || l != obj.info.ggml_nbytes()")
	}
	C.ggml_backend_tensor_set(tensor, unsafe.Pointer(unsafe.SliceData(data)), 0, C.size_t(l))
	return nil
}

func (obj Tensor) ForwardExpand() error {
	tensor, err := obj.check_ggml()
	if err != nil {
		return err
	}
	if obj.org._cgraph == nil {
		return errors.New("obj.org._cgraph == nil")
	}
	C.ggml_build_forward_expand(obj.org._cgraph, tensor)
	return nil
}

func (obj Tensor) Dup(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_DUP
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx])
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) SQR(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_SQR
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx])
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) SQRT(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_SQRT
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx])
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) LOG(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_LOG
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx])
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) SIN(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_SIN
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx])
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) COS(view bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_COS
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx])
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) Add(src Tensor, view bool) (Tensor, error) {
	if !src.info.P_can_repeat(obj.info) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1, obj1, err := src.org._tensors[src.info.idx], Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_ADD
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP, _tensor1)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx], _tensor1)
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) AddCast(src Tensor, t ggmlgo.GGML_TYPE) (Tensor, error) {
	if !LIB_ggml_is_quantized(obj.info.T) && obj.info.T != ggmlgo.GGML_TYPE_F16 && obj.info.T != ggmlgo.GGML_TYPE_BF16 {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if !src.info.P_can_repeat_rows(obj.info) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, _tensor1, obj1, err := obj.org._tensors[obj.info.idx], obj.org._tensors[src.info.idx], Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_ADD
	obj1.info, err = obj.org.ggml_new_tensor(t, obj.info.NE[:], obj1.info.OP, _tensor0, _tensor1)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) Add1(src Tensor, view bool) (Tensor, error) {
	if !src.info.P_is_scalar() {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
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
	_tensor1, obj1 := src.org._tensors[src.info.idx], Tensor{}
	obj1.info.OP = ggmlgo.GGML_OP_ADD1
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP, _tensor1)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, _tensor0, _tensor1)
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) Add_ID(src Tensor, id Tensor) (Tensor, error) {
	if id.info.T != ggmlgo.GGML_TYPE_I32 {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org || obj.org != id.org || src.org != id.org {
		return Tensor{}, errors.New("obj.org != src.org")
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
	_tensor0, _tensor1, _tensor2 := obj.org._tensors[obj.info.idx], src.org._tensors[src.info.idx], id.org._tensors[id.info.idx]
	if _tensor0.ne[0] != _tensor1.ne[0] {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if _tensor0.ne[1] != _tensor2.ne[0] {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if _tensor0.ne[2] != _tensor2.ne[1] {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], ggmlgo.GGML_OP_ADD_ID, _tensor0, _tensor1, _tensor2)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) ACC(src Tensor, view bool, params [4]int32) (Tensor, error) {
	if obj.info.T != ggmlgo.GGML_TYPE_F32 || src.info.T != ggmlgo.GGML_TYPE_F32 {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if !obj.info.P_is_contiguous() {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	n1, n2 := obj.info.ggml_nelements(), src.info.ggml_nelements()
	if n2 > n1 {
		return Tensor{}, errors.New("n2 > n1")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1 := src.org._tensors[src.info.idx]
	obj1, err := Tensor{}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_ACC
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP, _tensor1)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx], _tensor1)
	}
	if err == nil {
		obj1.org._tensors[obj1.info.idx].op_params[0] = C.int32_t(params[0])
		obj1.org._tensors[obj1.info.idx].op_params[1] = C.int32_t(params[1])
		obj1.org._tensors[obj1.info.idx].op_params[2] = C.int32_t(params[2])
		obj1.org._tensors[obj1.info.idx].op_params[3] = C.int32_t(params[3])
		obj1.org._tensors[obj1.info.idx].op_params[4] = 0
		if view {
			obj1.org._tensors[obj1.info.idx].op_params[4] = 1
		}
		obj1.info.OP, obj1.org = ggmlgo.GGML_OP_ACC, obj.org
	}
	return obj1, err
}

func (obj Tensor) SUB(src Tensor, view bool) (Tensor, error) {
	if !src.info.P_can_repeat(obj.info) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1, obj1, err := src.org._tensors[src.info.idx], Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_SUB
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP, _tensor1)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx], _tensor1)
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) MUL(src Tensor, view bool) (Tensor, error) {
	if !src.info.P_can_repeat(obj.info) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1, obj1, err := src.org._tensors[src.info.idx], Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_MUL
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP, _tensor1)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx], _tensor1)
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) DIV(src Tensor, view bool) (Tensor, error) {
	if !src.info.P_can_repeat(obj.info) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor1, obj1, err := src.org._tensors[src.info.idx], Tensor{org: obj.org}, error(nil)
	obj1.info.OP = ggmlgo.GGML_OP_DIV
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, obj1.info.OP, _tensor1)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], obj1.info.OP, obj.org._tensors[obj.info.idx], _tensor1)
	}
	if err != nil {
		return Tensor{}, err
	}
	return obj1, err
}

func (obj Tensor) SUM() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, b := obj.org._tensors[obj.info.idx], []int64{1}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, b, ggmlgo.GGML_OP_SUM, _tensor0)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) SUM_ROWS() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], ggmlgo.GGML_OP_SUM_ROWS, obj.org._tensors[obj.info.idx])
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) CUMSUM() (Tensor, error) {
	if obj.info.T != ggmlgo.GGML_TYPE_F32 {
		return Tensor{}, errors.New("obj.T != GGML_TYPE_F32")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], ggmlgo.GGML_OP_CUMSUM, obj.org._tensors[obj.info.idx])
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) MEAN() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, b := obj.org._tensors[obj.info.idx], obj.info.NE[:]
	obj1, err := Tensor{org: obj.org}, error(nil)
	b[0] = 1
	obj1.info, err = obj.org.ggml_new_tensor(ggmlgo.GGML_TYPE_F32, b, ggmlgo.GGML_OP_MEAN, _tensor0)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) ARGMAX() (Tensor, error) {
	if obj.info.NE[0] > 2147483647 {
		return Tensor{}, errors.New("> INT32_MAX")
	}
	if !obj.info.P_is_matrix() {
		return Tensor{}, errors.New("!obj.P_is_matrix()")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, b := obj.org._tensors[obj.info.idx], []int64{obj.info.NE[1]}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_new_tensor(ggmlgo.GGML_TYPE_I32, b, ggmlgo.GGML_OP_ARGMAX, _tensor0)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) COUNT_EQUAL(src Tensor) (Tensor, error) {
	if !obj.info.P_are_same_shape(src.info) {
		return Tensor{}, errors.New("!obj.P_are_same_shape()")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, _tensor1, b := obj.org._tensors[obj.info.idx], obj.org._tensors[src.info.idx], []int64{1}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_new_tensor(ggmlgo.GGML_TYPE_I64, b, ggmlgo.GGML_OP_COUNT_EQUAL, _tensor0, _tensor1)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) REPEAT(src Tensor) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	obj1, err := obj.REPEAT_4D(src.info.NE)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) REPEAT_4D(b1 [4]int64) (Tensor, error) {
	if !obj.info.P_is_empty() && !obj.info.P_can_repeat(TensorInfo{NE: b1}) {
		return Tensor{}, errors.New("!obj.P_is_empty() && !obj.P_can_repeat(src)")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, b := obj.org._tensors[obj.info.idx], b1[:]
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, b, ggmlgo.GGML_OP_REPEAT, _tensor0)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) REPEAT_BACK(src Tensor) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if !obj.info.P_can_repeat(src.info) {
		return Tensor{}, errors.New("!obj.P_can_repeat()")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, b := obj.org._tensors[obj.info.idx], src.info.NE[:]
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, b, ggmlgo.GGML_OP_REPEAT_BACK, _tensor0)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) MUL_MAT(src Tensor) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if obj.info.P_is_transposed() {
		return Tensor{}, errors.New("!obj.P_is_transposed()")
	}
	if !obj.info.P_can_mul_mat(src.info) {
		return Tensor{}, errors.New("!obj.P_can_mul_mat()")
	}

	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	_tensor0, _tensor1 := obj.org._tensors[obj.info.idx], obj.org._tensors[src.info.idx]
	b := []int64{obj.info.NE[1], src.info.NE[1], src.info.NE[2], src.info.NE[3]}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_new_tensor(ggmlgo.GGML_TYPE_F32, b, ggmlgo.GGML_OP_MUL_MAT, _tensor0, _tensor1)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}
