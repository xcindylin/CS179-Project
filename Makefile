CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /usr/local/cuda-6.5
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_35,code=sm_35

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifneq ($(DARWIN),)
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32 -rdc=true -lcudadevrt
else
      NVCCFLAGS := -m64 -rdc=true -lcudadevrt
endif

TARGETS = testgame
OBJS = board.o node.o exampleplayer.o player.o decisiontree.o gpuplayer.o paralleldecisiontree.o tree_cuda.o deviceboard.o devicenode.o

all: $(TARGETS)

gpuplayer.o: gpuplayer.cpp
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

paralleldecisiontree.o: paralleldecisiontree.cpp
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

tree_cuda.o: tree_cuda.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $^

deviceboard.o: deviceboard.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $^

devicenode.o: devicenode.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $^

testgame.o: testgame.cpp
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

link.o:
	$(NVCC) $(NVCCFLAGS) -dlink $(GENCODE_FLAGS) gpuplayer.o paralleldecisiontree.o tree_cuda.o deviceboard.o devicenode.o testgame.o -o $@

testgame: testgame.o $(OBJS) link.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LDFLAGS)

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
