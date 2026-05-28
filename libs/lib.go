// @@
// @ Author       : Eacher
// @ Date         : 2026-05-22 08:21:47
// @ LastEditTime : 2026-05-28 16:28:35
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

type ggml struct {
	_gctx             *C.struct_ggml_context
	_fctx             *C.struct_gguf_context
	_tensors          []*C.struct_ggml_tensor
	ctx               context.Context
	cancel            context.CancelCauseFunc
	is_init, is_close bool
}

func (gl *GGML) init(m Model) error {
	err := errors.New("is close or is init")
	if gl.org.is_init || gl.org.is_close {
		return err
	}
	if err = errors.New("not gguf"); gl.org._fctx == nil {
		return err
	}
	if err = errors.New("not ggml"); gl.org._gctx == nil {
		return err
	}
	gl.org.is_init = true
	m.Loader(gl.org)
	m.Devices(gl.org)
	m.LoadHparams(gl.org)
	m.LoadVocab(gl.org)
	m.LoadStatus(gl.org)
	m.LoadTensors(gl.org)
	gl.org.close()
	return nil
}

func (org *ggml) init(file string, no_alloc bool) error {
	if org.is_close || org._fctx != nil || org._gctx != nil {
		return errors.New("is close or is init")
	}
	name, params := C.CString(file), C.struct_gguf_init_params{no_alloc: C._Bool(no_alloc), ctx: &org._gctx}
	defer C.free(unsafe.Pointer(name))
	org._fctx = C.gguf_init_from_file(name, params)
	if org._fctx == nil {
		C.gguf_free(org._fctx)
		org._fctx = nil
		return errors.New("failed to load model from gguf")
	}
	if org._gctx == nil {
		C.ggml_free(org._gctx)
		org._gctx = nil
		return errors.New("failed to load model from ggml")
	}
	tensor := C.ggml_get_first_tensor(org._gctx)
	for tensor != nil {
		org._tensors = append(org._tensors, tensor)
		tensor = C.ggml_get_next_tensor(org._gctx, tensor)
	}
	return nil
}

func (org *ggml) done() {
	<-org.ctx.Done()
	org.close()
}

func (org *ggml) close() error {
	err := errors.New("is close")
	if !org.is_close {
		org.is_close, err = true, nil
		C.gguf_free(org._fctx)
		C.ggml_free(org._gctx)
		org._fctx, org._gctx = nil, nil
	}
	return err
}

func (org *ggml) ggml_new_tensor(t ggmlgo.GGML_TYPE, b []int64, op ggmlgo.GGML_OP, list ...*C.struct_ggml_tensor) (Tensor, error) {
	cur, err := Tensor{idx: len(org._tensors)}, errors.New("ggml is close")
	if org.is_close {
		return cur, err
	}
	n_dims := C.int(len(b))
	if n_dims < 1 || n_dims > 4 {
		return cur, errors.New("n_dims >= 1 && n_dims <= GGML_MAX_DIMS")
	}
	_tensor1 := C.ggml_new_tensor(org._gctx, C.enum_ggml_type(t), n_dims, (*C.int64_t)(unsafe.Pointer(&b)))
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

	cur.ne[0], cur.ne[1], cur.ne[2], cur.ne[3] = int64(_tensor1.ne[0]), int64(_tensor1.ne[1]), int64(_tensor1.ne[2]), int64(_tensor1.ne[3])
	cur.nb[0], cur.nb[1], cur.nb[2], cur.nb[3] = uint64(_tensor1.nb[0]), uint64(_tensor1.nb[1]), uint64(_tensor1.nb[2]), uint64(_tensor1.nb[3])
	cur.t, cur.op, cur.org = ggmlgo.GGML_TYPE(_tensor1._type), op, org
	return cur, nil
}

func (org *ggml) ggml_view_tensor(idx int, op ggmlgo.GGML_OP, list ...*C.struct_ggml_tensor) (Tensor, error) {
	cur, err := Tensor{idx: len(org._tensors)}, errors.New("ggml is close")
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
	cur.ne[0], cur.ne[1], cur.ne[2], cur.ne[3] = int64(_tensor1.ne[0]), int64(_tensor1.ne[1]), int64(_tensor1.ne[2]), int64(_tensor1.ne[3])
	cur.nb[0], cur.nb[1], cur.nb[2], cur.nb[3] = uint64(_tensor1.nb[0]), uint64(_tensor1.nb[1]), uint64(_tensor1.nb[2]), uint64(_tensor1.nb[3])
	cur.t, cur.op, cur.org, cur.name, cur.view = ggmlgo.GGML_TYPE(_tensor1._type), op, org, C.GoString(&_tensor1.name[0]), true
	return cur, nil
}
