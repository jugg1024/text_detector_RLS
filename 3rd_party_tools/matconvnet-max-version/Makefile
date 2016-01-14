SHELL=/bin/bash
#UNAME := $(shell uname)
#ifeq ($(UNAME),Darwin)
# MAC OSX
#	MEX  ?= /Applications/MATLAB_R2012b.app/bin/mex
#	MEXARCH = mexmaci64
#else
# TITAN
#	MEX  ?= /sync/matlab2013a/bin/mex
#	MEXARCH = mexa64
#endif

MEX  ?= /usr/local/MATLAB/R2014b/bin/mex
MEXARCH = mexa64
NVCC ?= /usr/local/cuda-7.0/bin/nvcc
ENABLE_GPU = true

NVCCOPTS = -gencode=arch=compute_20,code=sm_21 -gencode=arch=compute_30,code=sm_30 --compiler-options -fPIC

MEXOPTS = -lmwblas -largeArrayDims
MEXOPTS_GPU= $(MEXOPTS) -DENABLE_GPU -L /usr/local/cuda-7.0/targets/x86_64-linux/lib -lcudart -lcublas -lcudadevrt -f matlab/src/mex_CUDA_glnxa64.sh
ifneq ($(DEBUG),)
MEXOPTS += -g
endif
nvcc_filter=2> >(sed 's/^\(.*\)(\([0-9][0-9]*\)): \([ew].*\)/\1:\2: \3/g' >&2)

cpp_src:=matlab/src/bits/im2col.cpp
cpp_src+=matlab/src/bits/pooling.cpp
cpp_src+=matlab/src/bits/normalize.cpp

ifeq ($(ENABLE_GPU),FALSE)
mex_src:=matlab/src/gconv.cpp
mex_src+=matlab/src/gpool.cpp
mex_src+=matlab/src/gnormalize.cpp
mex_src+=matlab/src/gsepconv.cpp
mex_src+=matlab/src/gsepconv2.cpp
else
mex_src:=matlab/src/gconv.cu
mex_src+=matlab/src/gpool.cu
mex_src+=matlab/src/gnormalize.cu
mex_src+=matlab/src/gsepconv.cu
mex_src+=matlab/src/gsepconv2.cu
cpp_src+=matlab/src/bits/im2col_gpu.cu
cpp_src+=matlab/src/bits/pooling_gpu.cu
cpp_src+=matlab/src/bits/normalize_gpu.cu
endif

mex_tgt:=$(subst matlab/src/,matlab/mex/,$(mex_src))
mex_tgt:=$(patsubst %.cpp,%.$(MEXARCH),$(mex_tgt))
mex_tgt:=$(patsubst %.cu,%.$(MEXARCH),$(mex_tgt))

cpp_tgt:=$(patsubst %.cpp,%.o,$(cpp_src))
cpp_tgt:=$(patsubst %.cu,%.o,$(cpp_tgt))

.PHONY: all, distclean, clean, info

all: $(mex_tgt)

matlab/mex/.stamp:
	mkdir matlab/mex ; touch matlab/mex/.stamp

# Standard code
matlab/src/bits/%.o : matlab/src/bits/%.cpp
	$(MEX) -c $(MEXOPTS) "$(<)"
	mv -f "$(notdir $(@))" "$(@)"

matlab/src/bits/%.o : matlab/src/bits/%.cu
	$(NVCC) -dc $(NVCCOPTS) "$(<)" -o "$(@)" $(nvcc_filter)

matlab/src/bits/link.o : $(cpp_tgt)
	$(NVCC) -dlink $(NVCCOPTS) $(cpp_tgt) -o "$(@)" $(nvcc_filter)

# MEX files
matlab/mex/%.$(MEXARCH) : matlab/src/%.cpp matlab/mex/.stamp $(cpp_tgt)
	$(MEX) $(MEXOPTS) "$(<)" -o "$(@)" $(cu_tgt) $(nvcc_filter)

matlab/mex/%.$(MEXARCH) : matlab/src/%.cu matlab/mex/.stamp $(cpp_tgt) matlab/src/bits/link.o
ifeq ($(ENABLE_GPU),FALSE)
	echo "#include \"../src/$(notdir $(<))\"" > "matlab/mex/$(*).cpp"
	$(MEX) $(MEXOPTS) "matlab/mex/$(*).cpp" $(cpp_tgt) -o "$(@)" $(nvcc_filter)
	rm -f "matlab/mex/$(*).cpp"
else
	MW_NVCC_PATH='$(NVCC)' $(MEX) $(MEXOPTS_GPU) "matlab/src/bits/link.o" "$(<)" -o "$(@)" $(cpp_tgt) $(nvcc_filter)
endif

# Other targets
info:
	@echo "mex_src=$(mex_src)"
	@echo "mex_tgt=$(mex_tgt)"
	@echo "cpp_src=$(cpp_src)"
	@echo "cpp_tgt=$(cpp_tgt)"

clean:
	find . -name '*~' -delete
	rm -f $(cpp_tgt)

distclean: clean
	rm -rf matlab/mex/
