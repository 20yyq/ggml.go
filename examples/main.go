// @@
// @ Author       : Eacher
// @ Date         : 2023-09-13 16:48:07
// @ LastEditTime : 2026-06-29 14:11:56
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  :
// @ --------------------------------------------------------------------------------<
// @ FilePath     : /20yyq/test/go/main.go
// @@
package main

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"ggml.go/examples/simple"
)

var runMap map[string]func() error

func main() {
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGQUIT, syscall.SIGTERM)
	go func() {
		if len(os.Args) > 1 {
			if f, ok := runMap[os.Args[1]]; ok && f != nil {
				os.Args = append(os.Args[:1], os.Args[2:]...)
				if err := f(); err != nil {
					fmt.Println("method runing err:", err)
				}
				return
			}
		}
		fmt.Println("method not run")
		os.Exit(1)
	}()
	<-quit
	fmt.Println("End")
}

func init() {
	runMap = map[string]func() error{
		"simple": simple.Main,
	}
}
