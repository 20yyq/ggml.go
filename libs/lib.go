// @@
// @ Author       : Eacher
// @ Date         : 2026-05-22 08:21:47
// @ LastEditTime : 2026-06-24 09:58:04
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
	"unsafe"

	ggmlgo "ggml.go"
)

const GGML_MAX_DIMS = 4

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

	cur.NE[0], cur.NE[1], cur.NE[2], cur.NE[3] = int64(_tensor1.ne[0]), int64(_tensor1.ne[1]), int64(_tensor1.ne[2]), int64(_tensor1.ne[3])
	cur.NB[0], cur.NB[1], cur.NB[2], cur.NB[3] = uint64(_tensor1.nb[0]), uint64(_tensor1.nb[1]), uint64(_tensor1.nb[2]), uint64(_tensor1.nb[3])
	cur.T, cur.OP = ggmlgo.GGML_TYPE(_tensor1._type), op
	return cur, nil
}

func (org *GGML) ggml_view_tensor(idx int, op ggmlgo.GGML_OP, list ...*C.struct_ggml_tensor) (TensorInfo, error) {
	cur, err := TensorInfo{idx: len(org._tensors)}, errors.New("ggml is close")
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
	cur.NE[0], cur.NE[1], cur.NE[2], cur.NE[3] = int64(_tensor1.ne[0]), int64(_tensor1.ne[1]), int64(_tensor1.ne[2]), int64(_tensor1.ne[3])
	cur.NB[0], cur.NB[1], cur.NB[2], cur.NB[3] = uint64(_tensor1.nb[0]), uint64(_tensor1.nb[1]), uint64(_tensor1.nb[2]), uint64(_tensor1.nb[3])
	cur.T, cur.OP, cur.Name, cur.view = ggmlgo.GGML_TYPE(_tensor1._type), op, C.GoString(&_tensor1.name[0]), true
	return cur, nil
}
