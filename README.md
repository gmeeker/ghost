Ghost abstracts various GPU host APIs into a simple interface. Currently CUDA, OpenCL, and Metal are supported, as well as CPU.

This project aims to provide a set of functionality common to most APIs. It does not wrap every feature unique to an API.

- Load GPU libraries without a hard runtime dependency
- Compiling and executing kernels, and caching binaries
- Buffer and image management
- Data transfers between GPU and host
- Integration with your own GPU context
- Syncronization

Abstracting the kernel code is not addressed here. Using Slang is one possibility.
