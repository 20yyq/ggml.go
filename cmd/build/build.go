// @@
// @ Author       : Eacher
// @ Date         : 2026-05-24 10:36:25
// @ LastEditTime : 2026-05-24 16:02:06
// @ LastEditors  : Eacher
// @ --------------------------------------------------------------------------------<
// @ Description  : Please edit a descrition about this file at here.
// @ --------------------------------------------------------------------------------<
// @@
package main

import (
	"fmt"
	"os"
)

func main() {
	f, err := os.Create("../../libs/build/build.go")
	if err != nil {
		fmt.Println("Error creating:", err)
		os.Exit(1)
	}
	defer f.Close()
	b := `
//go:generate cmake ..
//go:generate cmake --build .
package build
`
	if _, err = f.WriteString(b); err != nil {
		fmt.Println("Error writing to:", err)
		os.Exit(1)
	}
}
