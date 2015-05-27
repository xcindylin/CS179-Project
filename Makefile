CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /usr/local/cuda-6.5
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

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
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

TARGETS = testgame
OBJS = testgame.o board.o deviceboard.o node.o devicenode.o exampleplayer.o player.o gpuplayer.o decisiontree.o paralleldecisiontree.o tree_cuda.o

all: $(TARGETS)

testgame: $(OBJS)
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ $^

gpuplayer.o: gpuplayer.cpp
	$(CC) -c -std=c++0x -O3 -Wall -I$(CUDA_INC_PATH) $<

paralleldecisiontree.o: paralleldecisiontree.cpp
	$(CC) -c -std=c++0x -O3 -Wall -I$(CUDA_INC_PATH) $<

tree_cuda.o: tree_cuda.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $^

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
