// @@
// @ Author       : Eacher
// @ Date         : 2026-05-22 08:21:47
// @ LastEditTime : 2026-05-25 17:53:10
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
// #include "expand.h"
//
// extern void go_log_callback(enum ggml_log_level level, char * text, void * user_data);
import "C"
import (
	"context"
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
	C.ggml_log_set(C.ggml_log_callback(C.go_log_callback), nil)
}

type ggml struct {
	_gctx             *C.struct_ggml_context
	_fctx             *C.struct_gguf_context
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
