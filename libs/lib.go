// @@
// @ Author       : Eacher
// @ Date         : 2026-05-22 08:21:47
// @ LastEditTime : 2026-05-27 14:08:21
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

func (org *ggml) ggml_new_tensor(t ggmlgo.GGML_TYPE, b []int64) (Tensor, error) {
	cur, err := Tensor{idx: len(org._tensors)}, errors.New("ggml is close")
	if org.is_close {
		return cur, err
	}
	n_dims := C.int(len(b))
	_tensor := C.ggml_new_tensor(org._gctx, C.enum_ggml_type(t), n_dims, (*C.int64_t)(unsafe.Pointer(&b)))
	if _tensor == nil {
		return cur, errors.New("ggml new _tensor err")
	}
	org._tensors = append(org._tensors, _tensor)
	cur.org, cur.T = org, t
	return cur, nil
}

func (org *ggml) ggml_view_tensor(idx int, list ...*C.struct_ggml_tensor) (Tensor, error) {
	cur, err := Tensor{idx: len(org._tensors)}, errors.New("ggml is close")
	if org.is_close {
		return cur, err
	}
	if !(idx < len(org._tensors)) || len(list) > 9 {
		return cur, errors.New("idx overris")
	}
	_tensor := C.ggml_view_tensor(org._gctx, org._tensors[idx])
	if _tensor == nil {
		return cur, errors.New("ggml new _tensor err")
	}
	org._tensors = append(org._tensors, _tensor)
	org._tensors[cur.idx].src[0] = _tensor
	i := 0
	for _, v := range list {
		org._tensors[cur.idx].src[i] = v
		i++
	}
	cur.org, cur.T, cur.Name = org, ggmlgo.GGML_TYPE(org._tensors[idx]._type), C.GoString(&_tensor.name[0])
	return cur, nil
}

func (org *ggml) ggml_dup_tensor(idx int, list ...*C.struct_ggml_tensor) (Tensor, error) {
	cur, err := Tensor{idx: len(org._tensors)}, errors.New("ggml is close")
	if org.is_close {
		return cur, err
	}
	if !(idx < len(org._tensors)) || len(list) > 9 {
		return cur, errors.New("idx overris")
	}
	_tensor := C.ggml_dup_tensor(org._gctx, org._tensors[idx])
	if _tensor == nil {
		return cur, errors.New("ggml new _tensor err")
	}
	org._tensors = append(org._tensors, _tensor)
	org._tensors[cur.idx].src[0] = _tensor
	i := 0
	for _, v := range list {
		org._tensors[cur.idx].src[i] = v
		i++
	}
	cur.org, cur.T, cur.Name = org, ggmlgo.GGML_TYPE(org._tensors[idx]._type), C.GoString(&_tensor.name[0])
	return cur, nil
}
