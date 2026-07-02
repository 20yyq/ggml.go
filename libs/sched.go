// @@
// @ Author       : Eacher
// @ Date         : 2026-06-27 15:24:40
// @ LastEditTime : 2026-06-30 10:29:12
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
	"io"
	"sync"
	"sync/atomic"
	"unsafe"

	ggmlgo "ggml.go"
)

type Sched struct {
	ctx               context.Context
	cancel            context.CancelCauseFunc
	is_init, is_close bool
}

func (ptr *Sched) Init(bs []*Backend, n_tensors uint64, ctx context.Context) error {
	err := ptr.init(bs, n_tensors)
	if err == nil {
		ptr.ctx, ptr.cancel = context.WithCancelCause(ctx)
		go ptr.done()
		for _, v := range bs {
			go func(v *Backend) {
				<-v.Done()
				ptr.cancel(io.EOF)
			}(v)
		}
	}
	return err
}

func (ptr *Sched) Done() <-chan struct{} {
	if ptr.ctx == nil {
		return nil
	}
	return ptr.ctx.Done()
}

func (ptr *Sched) Rest() error {
	if ptr.ctx == nil {
		return nil
	}
	return reset_sched(ptr)
}

func (ptr *Sched) AllocGraph(c *GGML) error {
	if ptr.ctx == nil {
		return nil
	}
	return alloc_graph(ptr, c._cgraph)
}

func (ptr *Sched) GraphCompute(c *GGML) (ResultTensor, error) {
	if ptr.ctx == nil {
		return ResultTensor{}, nil
	}
	return graph_compute(ptr, c._cgraph)
}

func (ptr *Sched) Close() error {
	if ptr.is_close {
		return errors.New("is close or is init")
	}
	if ptr.cancel != nil {
		ptr.cancel(io.EOF)
	}
	return nil
}

func (org *Sched) init(bs []*Backend, n_tensors uint64) error {
	err := errors.New("is close or is init")
	if org.is_init || org.is_close {
		return err
	}

	if err = sched(org, bs, n_tensors); err != nil {
		return err
	}
	org.is_init = true
	return nil
}

func (org *Sched) done() {
	<-org.ctx.Done()
	org.close()
}

func (org *Sched) close() error {
	err := errors.New("is close")
	if !org.is_close {
		org.is_close, err = true, nil
		delete_sched(org)
	}
	return err
}

// --------------------------------------------------

var scheds = struct {
	maps  sync.Map
	count atomic.Int64
	//
}{}

type sched_t struct {
	ptr    C.ggml_backend_sched_t
	lock   *sync.Mutex
	delete *bool
}

func sched(p *Sched, bs []*Backend, n_tensors uint64) error {
	if scheds.count.Load() > 15 {
		return errors.New("ptr.count.Load() > 16")
	}
	var l []C.ggml_backend_t
	var l1 []C.ggml_backend_buffer_type_t
	for _, v := range bs {
		if v.Check() == nil {
			if bptr := v.Dev.org.get_backend(v); bptr != nil {
				l = append(l, bptr)
				buf := C.ggml_backend_get_default_buffer_type(bptr)
				if info := v.Dev.org.info(); info.T == ggmlgo.GGML_BACKEND_DEVICE_TYPE_CPU {
					if tmp := C.ggml_backend_dev_host_buffer_type(v.Dev.org.ptr); tmp != nil {
						buf = tmp
					}
				}
				l1 = append(l1, buf)
			}
		}
	}
	if len(l) != len(l1) {
		return errors.New("ggml_backend_t != ggml_backend_buffer_type_t")
	}
	obj := sched_t{lock: &sync.Mutex{}, delete: new(bool)}
	bl := (*C.ggml_backend_t)(unsafe.Pointer(unsafe.SliceData(l)))
	bfl := (*C.ggml_backend_buffer_type_t)(unsafe.Pointer(unsafe.SliceData(l1)))
	n_tensors = max(2048, n_tensors*40)
	obj.ptr = C.ggml_backend_sched_new(bl, bfl, C.int(len(l)), C.size_t(n_tensors), false, true)
	if obj.ptr == nil {
		return errors.New("is close or is init")
	}
	if _, ok := scheds.maps.LoadOrStore(p, obj); ok {
		C.ggml_backend_sched_free(obj.ptr)

		return errors.New("is close or is init")
	}
	scheds.count.Add(1)
	return nil
}

func reset_sched(p *Sched) error {
	err := errors.New("is close or is init")
	obj := &sched_t{}
	scheds.maps.Range(func(key, value any) bool {
		if key.(*Sched) != p {
			return true
		}
		*obj = value.(sched_t)
		err = nil
		return false
	})
	if obj.lock != nil {
		obj.lock.Lock()
		if !*obj.delete {
			C.ggml_backend_sched_reset(obj.ptr)
		}
		obj.lock.Unlock()
	}
	return err
}

func alloc_graph(p *Sched, cgraph *C.struct_ggml_cgraph) error {
	err := errors.New("is close or is init")
	obj := &sched_t{}
	scheds.maps.Range(func(key, value any) bool {
		if key.(*Sched) != p {
			return true
		}
		*obj = value.(sched_t)
		err = nil
		return false
	})
	if obj.lock != nil {
		obj.lock.Lock()
		if !*obj.delete {
			C.ggml_backend_sched_alloc_graph(obj.ptr, cgraph)
		}
		obj.lock.Unlock()
	}
	return err
}

func graph_compute(p *Sched, cgraph *C.struct_ggml_cgraph) (ResultTensor, error) {
	err := errors.New("is close or is init")
	obj := &sched_t{}
	scheds.maps.Range(func(key, value any) bool {
		if key.(*Sched) != p {
			return true
		}
		*obj = value.(sched_t)
		err = nil
		return false
	})
	if obj.lock == nil {
		err = errors.New("is close or is init")
		return ResultTensor{}, err
	}
	obj.lock.Lock()
	if !*obj.delete {
		C.ggml_backend_sched_graph_compute(obj.ptr, cgraph)
	}
	obj.lock.Unlock()
	res := C.ggml_graph_node(cgraph, -1)
	var info ResultTensor
	info.Info.from_ggml_tensor(res)
	info.Data = make([]byte, info.Info.ggml_nbytes())
	C.ggml_backend_tensor_get(res, unsafe.Pointer(unsafe.SliceData(info.Data)), 0, C.size_t(len(info.Data)))
	return info, err
}

func delete_sched(p *Sched) {
	obj := &sched_t{}
	scheds.maps.Range(func(key, value any) bool {
		if key.(*Sched) != p {
			return true
		}
		*obj = value.(sched_t)
		return false
	})
	if obj.lock != nil {
		if _, ok := scheds.maps.LoadAndDelete(p); ok {
			scheds.count.Add(-1)
		}
		obj.lock.Lock()
		if !*obj.delete {
			*obj.delete = true
			C.ggml_backend_sched_free(obj.ptr)
		}
		obj.lock.Unlock()
	}
}
