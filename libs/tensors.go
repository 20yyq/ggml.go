// @@
// @ Author       : Eacher
// @ Date         : 2026-05-25 11:35:55
// @ LastEditTime : 2026-07-02 15:48:36
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

// #include "expand.h"
import "C"
import (
	"bytes"
	"encoding/binary"
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

func (obj TensorInfo) P_can_out_prod(obj1 TensorInfo) bool {
	// static inline bool ggml_can_out_prod(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
	//     static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

	//     return (t0->ne[1] == t1->ne[1])   &&
	//            (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
	//            (t1->ne[3]%t0->ne[3] == 0);
	// }
	return obj.NE[1] == obj1.NE[1] && (obj1.NE[2]%obj.NE[2] == 0) && (obj1.NE[3]%obj.NE[3] == 0)
}

func (obj TensorInfo) P_is_padded_1d() bool {
	// static inline bool ggml_is_padded_1d(const struct ggml_tensor * tensor) {
	// static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

	// 	return
	// 		tensor->nb[0] == ggml_type_size(tensor->type) &&
	// 		tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
	// 		tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
	// }
	return obj.NB[0] == LIB_ggml_type_size(obj.T) && obj.NB[2] == (obj.NB[1]*uint64(obj.NE[1])) && obj.NB[3] == (obj.NB[2]*uint64(obj.NE[2]))
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

// GGML_API struct ggml_tensor * ggml_dup
// GGML_API struct ggml_tensor * ggml_dup_inplace
func (obj Tensor) Dup(view bool) (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	var ptr *C.struct_ggml_tensor = nil
	if !view {
		ptr = C.ggml_dup(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_dup_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_add
// GGML_API struct ggml_tensor * ggml_add_inplace
func (obj Tensor) Add(src Tensor, view bool) (Tensor, error) {
	if !src.info.P_can_repeat(obj.info) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_add(obj.org._gctx, _tensor0, _tensor1)
	} else {
		ptr = C.ggml_add_inplace(obj.org._gctx, _tensor0, _tensor1)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_add_cast
func (obj Tensor) AddCast(src Tensor, t ggmlgo.GGML_TYPE) (Tensor, error) {
	if !LIB_ggml_is_quantized(obj.info.T) && obj.info.T != ggmlgo.GGML_TYPE_F16 && obj.info.T != ggmlgo.GGML_TYPE_BF16 {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if !src.info.P_can_repeat_rows(obj.info) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	if ptr = C.ggml_add_cast(obj.org._gctx, _tensor0, _tensor1, C.enum_ggml_type(t)); ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// dst[i0, i1, i2] = a[i0, i1, i2] + b[i0, ids[i1, i2]]
// GGML_API struct ggml_tensor * ggml_add_id
func (obj Tensor) Add_ID(src Tensor, id Tensor) (Tensor, error) {
	if id.info.T != ggmlgo.GGML_TYPE_I32 {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org || obj.org != id.org || src.org != id.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	var _tensor2 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _tensor2, err = id.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	if ptr = C.ggml_add_id(obj.org._gctx, _tensor0, _tensor1, _tensor2); ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// dst = a
// view(dst, nb1, nb2, nb3, offset) += b
// return dst
// GGML_API struct ggml_tensor * ggml_acc(
// GGML_API struct ggml_tensor * ggml_acc_inplace(
func (obj Tensor) ACC(src Tensor, view bool, params [4]uint64) (Tensor, error) {
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
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_acc(obj.org._gctx, _tensor0, _tensor1, C.size_t(params[0]), C.size_t(params[1]), C.size_t(params[2]), C.size_t(params[3]))
	} else {
		ptr = C.ggml_acc_inplace(obj.org._gctx, _tensor0, _tensor1, C.size_t(params[0]), C.size_t(params[1]), C.size_t(params[2]), C.size_t(params[3]))
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_sub(
// GGML_API struct ggml_tensor * ggml_sub_inplace(
func (obj Tensor) SUB(src Tensor, view bool) (Tensor, error) {
	if !src.info.P_can_repeat(obj.info) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_sub(obj.org._gctx, _tensor0, _tensor1)
	} else {
		ptr = C.ggml_sub_inplace(obj.org._gctx, _tensor0, _tensor1)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_mul(
// GGML_API struct ggml_tensor * ggml_mul_inplace(
func (obj Tensor) MUL(src Tensor, view bool) (Tensor, error) {
	if !src.info.P_can_repeat(obj.info) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_mul(obj.org._gctx, _tensor0, _tensor1)
	} else {
		ptr = C.ggml_mul_inplace(obj.org._gctx, _tensor0, _tensor1)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_div(
// GGML_API struct ggml_tensor * ggml_div_inplace(
func (obj Tensor) DIV(src Tensor, view bool) (Tensor, error) {
	if !src.info.P_can_repeat(obj.info) {
		return Tensor{}, errors.New("check if t1 can be represented as a repetition of t0")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_div(obj.org._gctx, _tensor0, _tensor1)
	} else {
		ptr = C.ggml_div_inplace(obj.org._gctx, _tensor0, _tensor1)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_sqr(
// GGML_API struct ggml_tensor * ggml_sqr_inplace(
func (obj Tensor) SQR(view bool) (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_sqr(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_sqr_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_sqrt(
// GGML_API struct ggml_tensor * ggml_sqrt_inplace(
func (obj Tensor) SQRT(view bool) (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_sqrt(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_sqrt_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_log(
// GGML_API struct ggml_tensor * ggml_log_inplace(
func (obj Tensor) LOG(view bool) (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_log(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_log_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_expm1(
// GGML_API struct ggml_tensor * ggml_expm1_inplace(
func (obj Tensor) EXPM1(view bool) (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_expm1(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_expm1_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_softplus(
// GGML_API struct ggml_tensor * ggml_softplus_inplace(
func (obj Tensor) SOFTPLUS(view bool) (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_softplus(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_softplus_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_sin(
// GGML_API struct ggml_tensor * ggml_sin_inplace(
func (obj Tensor) SIN(view bool) (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_sin(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_sin_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_cos(
// GGML_API struct ggml_tensor * ggml_cos_inplace(
func (obj Tensor) COS(view bool) (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_cos(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_cos_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// return scalar
// GGML_API struct ggml_tensor * ggml_sum(
func (obj Tensor) SUM() (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_sum(obj.org._gctx, _tensor0)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
// GGML_API struct ggml_tensor * ggml_sum_rows(
func (obj Tensor) SUM_ROWS() (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_sum_rows(obj.org._gctx, _tensor0)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_cumsum(
func (obj Tensor) CUMSUM() (Tensor, error) {
	if obj.info.T != ggmlgo.GGML_TYPE_F32 {
		return Tensor{}, errors.New("obj.T != GGML_TYPE_F32")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_cumsum(obj.org._gctx, _tensor0)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// mean along rows
// GGML_API struct ggml_tensor * ggml_mean(
func (obj Tensor) MEAN() (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_mean(obj.org._gctx, _tensor0)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// argmax along rows
// GGML_API struct ggml_tensor * ggml_argmax(
func (obj Tensor) ARGMAX() (Tensor, error) {
	if obj.info.NE[0] > 2147483647 {
		return Tensor{}, errors.New("> INT32_MAX")
	}
	if !obj.info.P_is_matrix() {
		return Tensor{}, errors.New("!obj.P_is_matrix()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_argmax(obj.org._gctx, _tensor0)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// count number of equal elements in a and b
// GGML_API struct ggml_tensor * ggml_count_equal(
func (obj Tensor) COUNT_EQUAL(src Tensor) (Tensor, error) {
	if !obj.info.P_are_same_shape(src.info) {
		return Tensor{}, errors.New("!obj.P_are_same_shape()")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_count_equal(obj.org._gctx, _tensor0, _tensor1)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// if a is the same shape as b, and a is not parameter, return a
// otherwise, return a new tensor: repeat(a) to fit in b
// GGML_API struct ggml_tensor * ggml_repeat(
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

// repeat a to the specified shape
// GGML_API struct ggml_tensor * ggml_repeat_4d(
func (obj Tensor) REPEAT_4D(b1 [4]int64) (Tensor, error) {
	if !obj.info.P_is_empty() && !obj.info.P_can_repeat(TensorInfo{NE: b1}) {
		return Tensor{}, errors.New("!obj.P_is_empty() && !obj.P_can_repeat(src)")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_repeat_4d(obj.org._gctx, _tensor0, C.int64_t(b1[0]), C.int64_t(b1[1]), C.int64_t(b1[2]), C.int64_t(b1[3]))
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// sums repetitions in a into shape of b
// GGML_API struct ggml_tensor * ggml_repeat_back(
func (obj Tensor) REPEAT_BACK(src Tensor) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if !obj.info.P_can_repeat(src.info) {
		return Tensor{}, errors.New("!obj.P_can_repeat()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_repeat_back(obj.org._gctx, _tensor0, _tensor1)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// concat a and b along dim
// used in stable-diffusion
// GGML_API struct ggml_tensor * ggml_concat(
func (obj Tensor) CONCAT(src Tensor, dim uint8) (Tensor, error) {
	if obj.org != src.org || obj.info.T != src.info.T || dim > 4 {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_concat(obj.org._gctx, _tensor0, _tensor1, C.int(dim))
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_abs(
// GGML_API struct ggml_tensor * ggml_abs_inplace(
func (obj Tensor) ABS(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_abs(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_abs_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_sgn(
// GGML_API struct ggml_tensor * ggml_sgn_inplace(
func (obj Tensor) SGN(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_sgn(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_sgn_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_neg(
// GGML_API struct ggml_tensor * ggml_neg_inplace(
func (obj Tensor) NEG(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_neg(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_neg_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_step(
// GGML_API struct ggml_tensor * ggml_step_inplace(
func (obj Tensor) STEP(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_step(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_step_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_tanh(
// GGML_API struct ggml_tensor * ggml_tanh_inplace(
func (obj Tensor) TANH(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_tanh(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_tanh_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_elu(
// GGML_API struct ggml_tensor * ggml_elu_inplace(
func (obj Tensor) ELU(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_elu(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_elu_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_relu(
// GGML_API struct ggml_tensor * ggml_relu_inplace(
func (obj Tensor) RELU(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_relu(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_relu_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_leaky_relu(
// func (obj Tensor) LEAKY_RELU(view bool, negative_slope float32) (Tensor, error) {
// 	if !obj.info.P_is_contiguous_rows() {
// 		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
// 	}
// 	_tensor0, err := obj.check_ggml()
// 	if err != nil {
// 		return Tensor{}, err
// 	}
// 	var ptr *C.struct_ggml_tensor = nil
// 	obj1 := Tensor{org: obj.org}
// 	ptr = C.ggml_leaky_relu(obj.org._gctx, _tensor0, C._Float)
// 	if ptr == nil {
// 		return Tensor{}, errors.New("tensor == nil")
// 	}
// 	obj1.org.push(ptr, &obj1.info)
// 	return obj1, nil
// }

// GGML_API struct ggml_tensor * ggml_sigmoid(
// GGML_API struct ggml_tensor * ggml_sigmoid_inplace(
func (obj Tensor) SIGMOID(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_sigmoid(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_sigmoid_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_gelu(
// GGML_API struct ggml_tensor * ggml_gelu_inplace(
func (obj Tensor) GELU(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_gelu(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_gelu_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_gelu_erf(
// GGML_API struct ggml_tensor * ggml_gelu_erf_inplace(
func (obj Tensor) GELU_ERF(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_gelu_erf(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_gelu_erf_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_gelu_quick(
// GGML_API struct ggml_tensor * ggml_gelu_quick_inplace(
func (obj Tensor) GELU_QUICK(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_gelu_quick(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_gelu_quick_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_silu(
// GGML_API struct ggml_tensor * ggml_silu_inplace(
func (obj Tensor) SILU(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_silu(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_silu_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// a - dy
// b - x
// GGML_API struct ggml_tensor * ggml_silu_back(
func (obj Tensor) SILU_BACK(src Tensor) (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_silu_back(obj.org._gctx, _tensor0, _tensor1)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// hardswish(x) = x * relu6(x + 3) / 6
// GGML_API struct ggml_tensor * ggml_hardswish(
func (obj Tensor) HARDSWISH() (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_hardswish(obj.org._gctx, _tensor0)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// hardsigmoid(x) = relu6(x + 3) / 6
// GGML_API struct ggml_tensor * ggml_hardsigmoid(
func (obj Tensor) HARDSIGMOID() (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_hardsigmoid(obj.org._gctx, _tensor0)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_exp(
// GGML_API struct ggml_tensor * ggml_exp_inplace(
func (obj Tensor) EXP(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_exp(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_exp_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_floor(
// GGML_API struct ggml_tensor * ggml_floor_inplace(
func (obj Tensor) FLOOR(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_floor(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_floor_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_ceil(
// GGML_API struct ggml_tensor * ggml_ceil_inplace(
func (obj Tensor) CEIL(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_ceil(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_ceil_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_round(
// GGML_API struct ggml_tensor * ggml_round_inplace(
func (obj Tensor) ROUND(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_round(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_round_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_trunc(
// GGML_API struct ggml_tensor * ggml_trunc_inplace(
func (obj Tensor) TRUNC(view bool) (Tensor, error) {
	if !obj.info.P_is_contiguous_rows() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_rows()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_trunc(obj.org._gctx, _tensor0)
	} else {
		ptr = C.ggml_trunc_inplace(obj.org._gctx, _tensor0)
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// ggml_leaky_relu

// struct ggml_tensor * ggml_leaky_relu(
// 	struct ggml_context * ctx,
// 	struct ggml_tensor  * a,
// 	float                 negative_slope,
// 	bool                  inplace) {
// struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

// ggml_set_op_params(result, &negative_slope, sizeof(negative_slope));

// result->op     = GGML_OP_LEAKY_RELU;
// result->src[0] = a;

// return result;
// }

// ggml_xielu

// struct ggml_tensor * ggml_xielu(
// 	struct ggml_context * ctx,
// 	struct ggml_tensor  * a,
// 	float alpha_n,
// 	float alpha_p,
// 	float beta,
// 	float eps) {
// struct ggml_tensor * result = ggml_dup_tensor(ctx, a);

// ggml_set_op_params_i32(result, 0, (int32_t) GGML_UNARY_OP_XIELU);
// ggml_set_op_params_f32(result, 1, beta + ggml_compute_softplus_f32(alpha_n));
// ggml_set_op_params_f32(result, 2, ggml_compute_softplus_f32(alpha_p));
// ggml_set_op_params_f32(result, 3, beta);
// ggml_set_op_params_f32(result, 4, eps);

// result->op     = GGML_OP_UNARY;
// result->src[0] = a;

// return result;
// }

// ggml_silu_back

// struct ggml_tensor * ggml_silu_back(
// 	struct ggml_context * ctx,
// 	struct ggml_tensor  * a,
// 	struct ggml_tensor  * b) {
// struct ggml_tensor * result = ggml_dup_tensor(ctx, a);

// result->op     = GGML_OP_SILU_BACK;
// result->src[0] = a;
// result->src[1] = b;

// return result;
// }

// ggml_glu

func (obj *Tensor) glu(src *Tensor, op []byte) (Tensor, error) {
	if !obj.info.P_is_contiguous_1() {
		return Tensor{}, errors.New("!obj.P_is_contiguous_1()")
	}
	list, b := []*C.struct_ggml_tensor{obj.org._tensors[obj.info.idx]}, obj.info.NE[:]
	b[0] = obj.info.NE[0] / 2
	if src != nil {
		b[0], list = obj.info.NE[0], append(list, obj.org._tensors[src.info.idx])
		if obj.info.T != src.info.T {
			return Tensor{}, errors.New("obj.info.T != src.info.T")
		}
		if !src.info.P_is_contiguous_1() {
			return Tensor{}, errors.New("!obj.P_is_contiguous_1()")
		}
		if !obj.info.P_are_same_shape(src.info) {
			return Tensor{}, errors.New("!obj.P_are_same_shape()")
		}
	}

	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, b, ggmlgo.GGML_OP_GLU, list...)

	if err != nil {
		return Tensor{}, err
	}
	set_op_params(obj1.org._tensors[obj1.info.idx], op)
	return obj1, nil
}

func (obj Tensor) GLU(op ggmlgo.GGML_GLU_OP, swapped bool) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(op), 0}
	if swapped {
		p[1] = 1
	}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) GLU_SPlIT(src Tensor, op ggmlgo.GGML_GLU_OP) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(op), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(&src, buf.Bytes())
}

// ggml_reglu

func (obj Tensor) REGLU() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_REGLU), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) REGLU_swapped() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_REGLU), 1}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) REGLU_split(src Tensor) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_REGLU), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(&src, buf.Bytes())
}

// ggml_geglu

func (obj Tensor) GEGLU() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_GEGLU), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) GEGLU_swapped() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_GEGLU), 1}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) GEGLU_split(src Tensor) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_GEGLU), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(&src, buf.Bytes())
}

// ggml_swiglu

func (obj Tensor) SWIGLU() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_SWIGLU), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) SWIGLU_swapped() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_SWIGLU), 1}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) SWIGLU_split(src Tensor) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_SWIGLU), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(&src, buf.Bytes())
}

// ggml_geglu_erf

func (obj Tensor) GEGLU_ERF() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_GEGLU_ERF), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) GEGLU_ERF_swapped() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_GEGLU_ERF), 1}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) GEGLU_ERF_split(src Tensor) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_GEGLU_ERF), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(&src, buf.Bytes())
}

// ggml_geglu_quick

func (obj Tensor) GEGLU_QUICK() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_GEGLU_QUICK), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) GEGLU_QUICK_swapped() (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_GEGLU_QUICK), 1}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(nil, buf.Bytes())
}

func (obj Tensor) GEGLU_QUICK_split(src Tensor) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []int32{int32(ggmlgo.GGML_GLU_OP_GEGLU_QUICK), 0}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(&src, buf.Bytes())
}

func (obj Tensor) SWIGLU_OAI(src Tensor, alpha, limit float32) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []float32{float32(ggmlgo.GGML_GLU_OP_SWIGLU_OAI), 0, alpha, limit}
	buf.Grow(16)
	binary.Write(buf, binary.LittleEndian, p)
	return obj.glu(&src, buf.Bytes())
}

// ggml_norm

func (obj Tensor) NORM(view bool, eps float32) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, ggmlgo.GGML_OP_NORM)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], ggmlgo.GGML_OP_NORM, obj.org._tensors[obj.info.idx])
	}
	if err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []float32{eps}
	buf.Grow(4)
	binary.Write(buf, binary.LittleEndian, p)
	set_op_params(obj1.org._tensors[obj1.info.idx], buf.Bytes())
	return obj1, nil
}

// ggml_rms_norm

func (obj Tensor) RMS_NORM(view bool, eps float32) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, ggmlgo.GGML_OP_RMS_NORM)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], ggmlgo.GGML_OP_RMS_NORM, obj.org._tensors[obj.info.idx])
	}
	if err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []float32{eps}
	buf.Grow(4)
	binary.Write(buf, binary.LittleEndian, p)
	set_op_params(obj1.org._tensors[obj1.info.idx], buf.Bytes())
	return obj1, nil
}

// ggml_rms_norm_back

func (obj Tensor) RMS_NORM_BACK(src Tensor, eps float32) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _, err := src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err, _tensor0, _tensor1 := Tensor{org: obj.org}, error(nil), obj.org._tensors[obj.info.idx], obj.org._tensors[src.info.idx]
	obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], ggmlgo.GGML_OP_RMS_NORM_BACK, _tensor0, _tensor1)
	if err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []float32{eps}
	buf.Grow(4)
	binary.Write(buf, binary.LittleEndian, p)
	set_op_params(obj1.org._tensors[obj1.info.idx], buf.Bytes())
	return obj1, nil
}

// ggml_group_norm

func (obj Tensor) GROUP_NORM(view bool, n_groups int32, eps float32) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, ggmlgo.GGML_OP_GROUP_NORM)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], ggmlgo.GGML_OP_GROUP_NORM, obj.org._tensors[obj.info.idx])
	}
	if err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []float32{float32(n_groups), eps}
	buf.Grow(8)
	binary.Write(buf, binary.LittleEndian, p)
	set_op_params(obj1.org._tensors[obj1.info.idx], buf.Bytes())
	return obj1, nil
}

// ggml_l2_norm

func (obj Tensor) L2_NORM(view bool, eps float32) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	if view {
		obj1.info, err = obj.org.ggml_view_tensor(obj.info.idx, ggmlgo.GGML_OP_L2_NORM)
	} else {
		obj1.info, err = obj.org.ggml_new_tensor(obj.info.T, obj.info.NE[:], ggmlgo.GGML_OP_L2_NORM, obj.org._tensors[obj.info.idx])
	}
	if err != nil {
		return Tensor{}, err
	}
	buf, p := bytes.NewBuffer(nil), []float32{eps}
	buf.Grow(4)
	binary.Write(buf, binary.LittleEndian, p)
	set_op_params(obj1.org._tensors[obj1.info.idx], buf.Bytes())
	return obj1, nil
}

// A: k columns, n rows => [ne03, ne02, n, k]
// B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
// result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
// GGML_API struct ggml_tensor * ggml_mul_mat(
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

	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_mul_mat(obj.org._gctx, _tensor0, _tensor1)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// void ggml_mul_mat_set_prec(
// 	struct ggml_tensor * a,
// 	enum ggml_prec       prec) {
// GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

// const int32_t prec_i32 = (int32_t) prec;

// ggml_set_op_params_i32(a, 0, prec_i32);
// }

// void ggml_mul_mat_set_hint(
// 	struct ggml_tensor * a,
// 	enum ggml_op_hint    hint) {
// GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

// const int32_t hint_i32 = (int32_t) hint;

// ggml_set_op_params_i32(a, 1, hint_i32);
// }

/*
	c = ggml_mul_mat_id(ctx, as, b, ids);

	as  -> [cols, rows, n_expert]
	b   -> [cols, n_expert_used, n_tokens]
	ids -> [n_expert_used, n_tokens] (i32)
	c   -> [rows, n_expert_used, n_tokens]

	in b, n_expert_used can be broadcasted to match the n_expert_used of ids

	c ~= as[:,:,i] @ b[:,i%r,t], i = ids[e,t] for all e,t in ids
*/
// indirect matrix multiplication
// GGML_API struct ggml_tensor * ggml_mul_mat_id(
func (obj Tensor) MUL_MAT_ID(src, ids Tensor) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if ids.info.T != ggmlgo.GGML_TYPE_I32 {
		return Tensor{}, errors.New("ids.info.T != ggmlgo.GGML_TYPE_I32")
	}
	// GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
	// GGML_ASSERT(b->ne[3] == 1); // b is 3d
	// GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
	// GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
	// GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
	// GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast
	if obj.info.NE[3] != 1 || // as is 3d (one matrix per expert)
		src.info.NE[3] != 1 || // b is 3d
		ids.info.NE[2] != 1 || ids.info.NE[3] != 1 || // ids is 2d
		ids.info.NE[1] != src.info.NE[2] || //  must have an expert list per b row
		obj.info.NE[0] != src.info.NE[0] || // can_mul_mat
		ids.info.NE[0]%src.info.NE[1] != 0 { // can broadcast
		return Tensor{}, errors.New("ids.info.T != ggmlgo.GGML_TYPE_I32")
	}

	if obj.info.P_is_transposed() {
		return Tensor{}, errors.New("!obj.P_is_transposed()")
	}

	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	var _tensor2 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	if _tensor2, err = ids.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_mul_mat_id(obj.org._gctx, _tensor0, _tensor1, _tensor2)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// A: m columns, n rows,
// B: p columns, n rows,
// result is m columns, p rows
// GGML_API struct ggml_tensor * ggml_out_prod(
func (obj Tensor) OUT_PROD(src Tensor) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if obj.info.P_is_transposed() {
		return Tensor{}, errors.New("!obj.P_is_transposed()")
	}
	if !obj.info.P_can_out_prod(src.info) {
		return Tensor{}, errors.New("!obj.P_can_out_prod()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_out_prod(obj.org._gctx, _tensor0, _tensor1)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_scale(
// in-place, returns view(a)
// GGML_API struct ggml_tensor * ggml_scale_inplace(
// x = s * a + b
// GGML_API struct ggml_tensor * ggml_scale_bias(
// GGML_API struct ggml_tensor * ggml_scale_bias_inplace(
func (obj Tensor) SCALE(view bool, f []float32) (Tensor, error) {
	if !obj.info.P_is_padded_1d() {
		return Tensor{}, errors.New("!obj.P_is_padded_1d()")
	}
	obj1, l := Tensor{org: obj.org}, len(f)
	if l < 1 || l > 2 {
		return Tensor{}, errors.New("l < 1 || l > 2")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	switch l {
	case 1:
		if !view {
			ptr = C.ggml_scale(obj.org._gctx, _tensor0, C.float(f[0]))
		} else {
			ptr = C.ggml_scale_inplace(obj.org._gctx, _tensor0, C.float(f[0]))
		}
	case 2:
		if !view {
			ptr = C.ggml_scale_bias(obj.org._gctx, _tensor0, C.float(f[0]), C.float(f[1]))
		} else {
			ptr = C.ggml_scale_bias_inplace(obj.org._gctx, _tensor0, C.float(f[0]), C.float(f[1]))
		}
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// b -> view(a,offset,nb1,nb2,3), return modified a
// GGML_API struct ggml_tensor * ggml_set(
// b -> view(a,offset,nb1,nb2,3), return view(a)
// GGML_API struct ggml_tensor * ggml_set_inplace(
func (obj Tensor) SET(src Tensor, view bool, eps [4]uint64) (Tensor, error) {
	if eps[3] >= uint64(1<<30) {
		return Tensor{}, errors.New(">= 1<<30")
	}
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if src.info.ggml_nelements() > obj.info.ggml_nelements() {
		return Tensor{}, errors.New("src.info.ggml_nelements() > obj.info.ggml_nelements()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	if !view {
		ptr = C.ggml_set(obj.org._gctx, _tensor0, _tensor1, C.size_t(eps[0]), C.size_t(eps[1]), C.size_t(eps[2]), C.size_t(eps[3]))
	} else {
		ptr = C.ggml_set_inplace(obj.org._gctx, _tensor0, _tensor1, C.size_t(eps[0]), C.size_t(eps[1]), C.size_t(eps[2]), C.size_t(eps[3]))
	}
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_set_1d(
// GGML_API struct ggml_tensor * ggml_set_1d_inplace(
func (obj Tensor) SET_1D(src Tensor, view bool, offset uint64) (Tensor, error) {
	eps := [4]uint64{obj.info.NB[1], obj.info.NB[2], obj.info.NB[3], offset}
	return obj.SET(src, view, eps)
}

// b -> view(a,offset,nb1,nb2,3), return modified a
// GGML_API struct ggml_tensor * ggml_set_2d(
// b -> view(a,offset,nb1,nb2,3), return view(a)
// GGML_API struct ggml_tensor * ggml_set_2d_inplace(
func (obj Tensor) SET_2D(src Tensor, view bool, nb1, offset uint64) (Tensor, error) {
	eps := [4]uint64{nb1, obj.info.NB[2], obj.info.NB[3], offset}
	return obj.SET(src, view, eps)
}

// a -> b, return view(b)
// GGML_API struct ggml_tensor * ggml_cpy(
func (obj Tensor) CPY(src Tensor) (Tensor, error) {
	if obj.org != src.org {
		return Tensor{}, errors.New("obj.org != src.org")
	}
	if src.info.ggml_nelements() != obj.info.ggml_nelements() {
		return Tensor{}, errors.New("src.info.ggml_nelements() != obj.info.ggml_nelements()")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	var _tensor1 *C.struct_ggml_tensor = nil
	if _tensor1, err = src.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_cpy(obj.org._gctx, _tensor0, _tensor1)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// note: casting from f32 to i32 will discard the fractional part
// GGML_API struct ggml_tensor * ggml_cast(
func (obj Tensor) CAST(t ggmlgo.GGML_TYPE) (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_cast(obj.org._gctx, _tensor0, C.enum_ggml_type(t))
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// make contiguous
// GGML_API struct ggml_tensor * ggml_cont(
func (obj Tensor) CONT() (Tensor, error) {
	count := int64(1)
	for _, v := range obj.info.NE {
		count *= v
	}
	if obj.info.ggml_nelements() != count {
		return Tensor{}, errors.New("obj.info.ggml_nelements() != count ")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_cont(obj.org._gctx, _tensor0)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// make contiguous, with new shape
// GGML_API struct ggml_tensor * ggml_cont_1d(
func (obj Tensor) CONT_1D(d int64) (Tensor, error) {
	n := [4]int64{d, 1, 1, 1}
	return obj.CONT_4D(n)
}

// GGML_API struct ggml_tensor * ggml_cont_2d(
func (obj Tensor) CONT_2D(d [2]int64) (Tensor, error) {
	n := [4]int64{d[0], d[1], 1, 1}
	return obj.CONT_4D(n)
}

// GGML_API struct ggml_tensor * ggml_cont_3d(
func (obj Tensor) CONT_3D(d [3]int64) (Tensor, error) {
	n := [4]int64{d[0], d[1], d[2], 1}
	return obj.CONT_4D(n)
}

// GGML_API struct ggml_tensor * ggml_cont_4d(
func (obj Tensor) CONT_4D(d [4]int64) (Tensor, error) {
	count := int64(1)
	for _, v := range obj.info.NE {
		count *= v
	}
	if obj.info.ggml_nelements() != count {
		return Tensor{}, errors.New("obj.info.ggml_nelements() != count ")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_cont_4d(obj.org._gctx, _tensor0, C.int64_t(d[0]), C.int64_t(d[1]), C.int64_t(d[2]), C.int64_t(d[3]))
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// ggml_reshape

func (obj *Tensor) reshape(b []int64) (Tensor, error) {
	if !obj.info.P_is_contiguous() {
		return Tensor{}, errors.New("!obj.P_is_contiguous()")
	}
	count := int64(1)
	obj1, err, info := Tensor{org: obj.org}, error(nil), TensorInfo{NE: [4]int64{1, 1, 1, 1}}
	for k, v := range b {
		count *= v
		info.NE[k] = v
	}
	if obj.info.ggml_nelements() != count {
		return Tensor{}, errors.New("obj.info.ggml_nelements() != count ")
	}
	obj1.info, err = obj.org.ggml_reshape(obj.info.idx, &info)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

func (obj Tensor) RESHAPE(src Tensor) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	return obj.reshape(src.info.NE[:])
}

func (obj Tensor) RESHAPE_1D(d int64) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	return obj.reshape([]int64{d})
}

func (obj Tensor) RESHAPE_2D(d [2]int64) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	return obj.reshape(d[:])
}

func (obj Tensor) RESHAPE_3D(d [3]int64) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	return obj.reshape(d[:])
}

func (obj Tensor) RESHAPE_4D(d [4]int64) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	return obj.reshape(d[:])
}

// ggml_view_1d
// ggml_view_2d
// ggml_view_3d
// ggml_view_4d

func (obj Tensor) VIEW_D(ne []int64, nb []uint64) (Tensor, error) {
	if _, err := obj.check_ggml(); err != nil {
		return Tensor{}, err
	}
	obj1, err := Tensor{org: obj.org}, error(nil)
	obj1.info, err = obj.org.ggml_view_d(obj.info.idx, ne, nb)
	if err != nil {
		return Tensor{}, err
	}
	return obj1, nil
}

// GGML_API struct ggml_tensor * ggml_permute(
func (obj Tensor) PERMUTE(axis0, axis1, axis2, axis3 uint8) (Tensor, error) {
	if axis0 > 4 || axis1 > 4 || axis2 > 4 || axis3 > 4 {
		return Tensor{}, errors.New("axis0 > 4 || axis1 > 4 || axis2 > 4 || axis3 > 4")
	}
	if axis0 == axis1 || axis0 == axis2 || axis0 == axis3 {
		return Tensor{}, errors.New("axis0 > 4 || axis1 > 4 || axis2 > 4 || axis3 > 4")
	}
	if axis1 == axis2 || axis1 == axis3 || axis2 == axis3 {
		return Tensor{}, errors.New("axis0 > 4 || axis1 > 4 || axis2 > 4 || axis3 > 4")
	}
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_permute(obj.org._gctx, _tensor0, C.int(axis0), C.int(axis1), C.int(axis2), C.int(axis3))
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}

// alias for ggml_permute(ctx, a, 1, 0, 2, 3)
// GGML_API struct ggml_tensor * ggml_transpose(
func (obj Tensor) TRANSPOSE() (Tensor, error) {
	_tensor0, err := obj.check_ggml()
	if err != nil {
		return Tensor{}, err
	}
	var ptr *C.struct_ggml_tensor = nil
	obj1 := Tensor{org: obj.org}
	ptr = C.ggml_transpose(obj.org._gctx, _tensor0)
	if ptr == nil {
		return Tensor{}, errors.New("tensor == nil")
	}
	obj1.org.push(ptr, &obj1.info)
	return obj1, nil
}
