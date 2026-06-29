// @@
// @ Author       : Eacher
// @ Date         : 2026-05-23 22:16:49
// @ LastEditTime : 2026-06-29 15:00:21
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package libs

// #include "expand.h"
import "C"
import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"

	ggmlgo "ggml.go"
)

type DevInfo struct {
	T                        ggmlgo.GGML_BACKEND_DEV_TYPE
	IsNuma                   bool
	DevName, DevDes, RegName string
	MemoryFree, MemoryTotal  uint64 // device free memory in bytes device total memory in bytes
}

type dev struct {
	ptr     C.ggml_backend_dev_t
	reg     C.ggml_backend_reg_t
	props   C.struct_ggml_backend_dev_props
	is_numa bool
	childes sync.Map
	count   atomic.Int64
}

var devs []*dev

func backend_init(is bool) {
	// needed to initialize f16 tables
	C.ggml_free(C.ggml_init(C.struct_ggml_init_params{}))

	if C.ggml_backend_load_all(); LIB_reg_count() == 0 {
		// hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function
		panic("no backends are loaded.")
	}

	for i := uint64(0); i < LIB_dev_count(); i++ {
		ptr := &dev{ptr: C.ggml_backend_dev_get(C.size_t(i))}
		ptr.reg = C.ggml_backend_dev_backend_reg(ptr.ptr)
		C.ggml_backend_dev_get_props(ptr.ptr, &ptr.props)
		if ggmlgo.GGML_BACKEND_DEV_TYPE(ptr.props._type) == ggmlgo.GGML_BACKEND_DEVICE_TYPE_CPU && is {
			C.numa_init_fn(ptr.reg, C.GGML_NUMA_STRATEGY_NUMACTL)
			ptr.is_numa = bool(C.cpu_is_numa(ptr.reg))
		}
		ptr.childes.Clear()
		devs = append(devs, ptr)
	}
}

func backend_dinit() {
	for _, value := range devs {
		value.childes.Range(func(key, value any) bool {
			key.(*Backend).cancel(io.EOF)
			return true
		})
		value.childes.Clear()
	}
	devs = nil
	C.ggml_quantize_free()
	fmt.Printf("backend_dinit \n")
}

func (ptr *dev) backend(p *Backend) error {
	if ptr.count.Load() > 15 {
		return errors.New("ptr.count.Load() > 16")
	}
	backend := C.ggml_backend_dev_init(ptr.ptr, nil)
	if backend == nil {
		return errors.New("is close or is init")
	}
	if _, ok := ptr.childes.LoadOrStore(p, backend); ok {
		C.ggml_backend_free(backend)
		return errors.New("is close or is init")
	}
	ptr.count.Add(1)
	return nil
}

func (ptr *dev) delete_backend(p *Backend) {
	value, ok := ptr.childes.LoadAndDelete(p)
	if ok {
		ptr.count.Add(-1)
		C.ggml_backend_free(value.(C.ggml_backend_t))
	}
}

func (ptr *dev) get_backend(p *Backend) C.ggml_backend_t {
	var bptr C.ggml_backend_t
	ptr.childes.Range(func(key, value any) bool {
		if key.(*Backend) == p {
			bptr = value.(C.ggml_backend_t)
			return false
		}
		return true
	})
	return bptr
}
func (ptr *dev) info() DevInfo {
	var info DevInfo
	info.T, info.IsNuma = ggmlgo.GGML_BACKEND_DEV_TYPE(ptr.props._type), ptr.is_numa
	info.DevName, info.DevDes = C.GoString(ptr.props.name), C.GoString(ptr.props.description)
	info.RegName = C.GoString(C.ggml_backend_reg_name(ptr.reg))
	info.MemoryFree, info.MemoryTotal = uint64(ptr.props.memory_free), uint64(ptr.props.memory_total)
	return info
}

type Dev struct {
	org  *dev
	idx  uint8
	info DevInfo
}

func (ptr *Dev) Info() DevInfo {
	return ptr.info
}

func (ptr *Dev) Set_n_threads(n uint16) error {

	return nil
}

type Backend struct {
	ctx               context.Context
	cancel            context.CancelCauseFunc
	is_init, is_close bool

	Dev Dev
}

func (ptr *Backend) Init(ctx context.Context) error {
	err := ptr.init()
	if err == nil {
		ptr.ctx, ptr.cancel = context.WithCancelCause(ctx)
		go ptr.done()
	}
	return err
}

func (ptr *Backend) Done() <-chan struct{} {
	if ptr.ctx == nil {
		return nil
	}
	return ptr.ctx.Done()
}

func (ptr *Backend) Close() error {
	if ptr.is_close {
		return errors.New("is close or is init")
	}
	if ptr.cancel != nil {
		ptr.cancel(io.EOF)
	}
	return nil
}

func (org *Backend) init() error {
	err := errors.New("is close or is init")
	if org.Dev.org == nil || org.is_init || org.is_close {
		return err
	}

	if err = org.Dev.org.backend(org); err != nil {
		return err
	}
	org.is_init = true
	return nil
}

func (org *Backend) done() {
	<-org.ctx.Done()
	org.close()
}

func (org *Backend) close() error {
	err := errors.New("is close")
	if !org.is_close {
		org.Dev.org.delete_backend(org)
		org.is_close, err = true, nil
	}
	return err
}
