// @@
// @ Author       : Eacher
// @ Date         : 2026-05-22 08:21:47
// @ LastEditTime : 2026-07-02 11:33:14
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

// #include "expand.h"
//
import "C"
import (
	"context"
	"errors"
	"fmt"
	"io"
	"unsafe"

	ggmlgo "ggml.go"
)

//export go_log_callback
func go_log_callback(level C.enum_ggml_log_level, text *C.char, _ unsafe.Pointer) {
	fmt.Printf("%d %s", level, C.GoString(text))
}

//export go_abort_callback
func go_abort_callback(text *C.char) {
	panic(C.GoString(text))
}

func init() {
	C.ggml_log_set(C.ggml_log_callback(C.go_log_callback), nil)
	C.ggml_set_abort_callback(C.ggml_abort_callback_t(C.go_abort_callback))
}

type GGML struct {
	_gctx                       *C.struct_ggml_context
	_cgraph                     *C.struct_ggml_cgraph
	_tensors                    []*C.struct_ggml_tensor
	data                        []byte
	ctx                         context.Context
	cancel                      context.CancelCauseFunc
	is_init, is_close, is_graph bool
}

func (ptr *GGML) Init(n uint64, cgraph bool, ctx context.Context) error {
	err := ptr.init(n, cgraph)
	if err == nil {
		ptr.ctx, ptr.cancel = context.WithCancelCause(ctx)
		go ptr.done()
	}
	return err
}

func (ptr *GGML) Close() error {
	if !ptr.is_init {
		return errors.New("is close or is init")
	}
	ptr.cancel(io.EOF)
	return nil
}

func (org *GGML) init(n_tensors uint64, is_graph bool) error {
	if org.is_init {
		return errors.New("is close or is init")
	}
	gn := LIB_ggml_padding(uint64(unsafe.Sizeof(C._c_ggml_context_t{})), 16)
	n := gn + n_tensors*LIB_ggml_tensor_overhead()
	if org.is_graph = is_graph; is_graph {
		n += LIB_ggml_graph_overhead_custom(n_tensors)
	}
	n = LIB_ggml_padding(n, 16)
	org.data = make([]byte, n)
	if org.data == nil {
		return errors.New("overflow byte")
	}
	org.is_init = true
	ptr := unsafe.Pointer(unsafe.SliceData(org.data))
	org._gctx = (*C.struct_ggml_context)(ptr)
	*(*uint64)(ptr) = n - gn
	ptr = unsafe.Pointer(uintptr(ptr) + unsafe.Sizeof(n))
	*(*unsafe.Pointer)(ptr) = unsafe.Pointer(unsafe.SliceData(org.data[gn:]))
	ptr = unsafe.Pointer(uintptr(ptr) + unsafe.Sizeof(ptr))
	*(*bool)(ptr) = false
	ptr = unsafe.Pointer(uintptr(ptr) + unsafe.Sizeof(false))
	*(*bool)(ptr) = true
	if is_graph {
		org._cgraph = C.ggml_new_graph_custom(org._gctx, C.size_t(n_tensors), false)
	}
	return nil
}

func (org *GGML) done() {
	<-org.ctx.Done()
	org.close()
}

func (org *GGML) close() error {
	err := errors.New("is close")
	// C.ggml_print_objects(org._gctx)
	if !org.is_close {
		org.is_close, err = true, nil
		org._cgraph, org._gctx = nil, nil
	}
	return err
}

func (org *GGML) push(ptr *C.struct_ggml_tensor, info *TensorInfo) {
	info.idx = len(org._tensors)
	org._tensors = append(org._tensors, ptr)
	info.from_ggml_tensor(ptr)
}

func (org *GGML) ggml_new_tensor(t ggmlgo.GGML_TYPE, b []int64, op ggmlgo.GGML_OP, list ...*C.struct_ggml_tensor) (TensorInfo, error) {
	cur, err := TensorInfo{idx: len(org._tensors)}, errors.New("ggml is close")
	if org.is_close {
		return cur, err
	}
	if len(list) > 10 {
		return cur, errors.New("idx overris")
	}
	n_dims := C.int(len(b))
	if n_dims < 1 || n_dims > 4 {
		return cur, errors.New("n_dims >= 1 && n_dims <= GGML_MAX_DIMS")
	}
	_tensor1 := C.ggml_new_tensor(org._gctx, C.enum_ggml_type(t), n_dims, (*C.int64_t)(unsafe.SliceData(b)))
	if _tensor1 == nil {
		return cur, errors.New("ggml new _tensor err")
	}
	org._tensors = append(org._tensors, _tensor1)
	i := 0
	for _, v := range list {
		_tensor1.src[i] = v
		i++
	}
	_tensor1.op = C.enum_ggml_op(op)
	cur.from_ggml_tensor(_tensor1)
	return cur, nil
}

func (org *GGML) ggml_view_tensor(idx int, op ggmlgo.GGML_OP, list ...*C.struct_ggml_tensor) (TensorInfo, error) {
	cur, err := TensorInfo{idx: len(org._tensors), view: true}, errors.New("ggml is close")
	if org.is_close {
		return cur, err
	}
	if !(idx < len(org._tensors)) || len(list) > 9 {
		return cur, errors.New("idx overris")
	}
	_tensor0 := org._tensors[idx]
	_tensor1 := C.ggml_view_tensor(org._gctx, _tensor0)
	if _tensor1 == nil {
		return cur, errors.New("ggml new _tensor err")
	}
	list = append([]*C.struct_ggml_tensor{_tensor0}, list...)
	org._tensors = append(org._tensors, _tensor1)
	i := 0
	for _, v := range list {
		_tensor1.src[i] = v
		i++
	}
	_tensor1.op = C.enum_ggml_op(op)
	cur.from_ggml_tensor(_tensor1)
	return cur, nil
}

func (org *GGML) ggml_reshape(idx int, info *TensorInfo) (TensorInfo, error) {
	cur, err := TensorInfo{idx: len(org._tensors)}, errors.New("ggml is close")
	if org.is_close {
		return cur, err
	}
	if !(idx < len(org._tensors)) {
		return cur, errors.New("idx overris")
	}
	_tensor0 := org._tensors[idx]
	var _tensor0_1 C.struct_ggml_tensor
	var ne []int64
	ne = unsafe.Slice((*int64)(unsafe.Pointer(&_tensor0_1.ne[0])), 4)
	copy(ne, info.NE[:])
	_tensor1 := C.ggml_reshape(org._gctx, _tensor0, &_tensor0_1)
	if _tensor1 == nil {
		return cur, errors.New("ggml new _tensor err")
	}
	cur.from_ggml_tensor(_tensor1)
	return cur, nil
}

func (org *GGML) ggml_view_d(idx int, ne []int64, nb []uint64) (TensorInfo, error) {
	cur, err := TensorInfo{idx: len(org._tensors), view: true}, errors.New("ggml is close")
	if org.is_close {
		return cur, err
	}
	l := len(ne)
	if l < 1 || l > 4 || l != len(nb) || !(idx < len(org._tensors)) {
		return cur, errors.New("idx overris")
	}
	_tensor0 := org._tensors[idx]
	var _tensor1 *C.struct_ggml_tensor = nil
	switch l {
	case 1:
		_tensor1 = C.ggml_view_1d(org._gctx, _tensor0, C.int64_t(ne[0]), C.size_t(nb[0]))
	case 2:
		_tensor1 = C.ggml_view_2d(org._gctx, _tensor0, C.int64_t(ne[0]), C.int64_t(ne[1]), C.size_t(nb[0]), C.size_t(nb[1]))
	case 3:
		_tensor1 = C.ggml_view_3d(org._gctx, _tensor0, C.int64_t(ne[0]), C.int64_t(ne[1]), C.int64_t(ne[2]), C.size_t(nb[0]), C.size_t(nb[1]), C.size_t(nb[2]))
	case 4:
		_tensor1 = C.ggml_view_4d(org._gctx, _tensor0, C.int64_t(ne[0]), C.int64_t(ne[1]), C.int64_t(ne[2]), C.int64_t(ne[3]), C.size_t(nb[0]), C.size_t(nb[1]), C.size_t(nb[2]), C.size_t(nb[3]))
	}
	if _tensor1 == nil {
		return cur, errors.New("ggml new _tensor err")
	}
	cur.from_ggml_tensor(_tensor1)
	return cur, nil
}

func set_op_params(tensor *C.struct_ggml_tensor, b []byte) error {
	if len(b) > 64 {
		return errors.New("params_size > GGML_MAX_OP_PARAMS")
	}
	var op_params []byte
	op_params = unsafe.Slice((*byte)(unsafe.Pointer(&tensor.op_params[0])), 64)
	copy(op_params, b)
	return nil
}
