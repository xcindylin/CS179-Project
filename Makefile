CC = /usr/bin/g++

LD_FLAGS = -lrt -dc

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

all: $(TARGETS)

testgame: testgame.cpp testgame.o board.o deviceboard.o decisiontree.o paralleldecisiontree.o node.o devicenode.o exampleplayer.o player.o gpuplayer.o
	$(CC) $< -std=c++0x -o $@ testgame.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

testgame.o: tree_cuda.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

exampleplayer.o: exampleplayer.cpp
	$(CC) $< -std=c++0x -o $@ testgame.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

player.o: player.cpp
	$(CC) $< -std=c++0x -o $@ testgame.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

gpuplayer.o: gpuplayer.cpp
	$(CC) $< -std=c++0x -o $@ testgame.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

devicenode.o: devicenode.cpp
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

node.o: node.cpp
	$(CC) $< -std=c++0x -o $@ testgame.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

paralleldecisiontree.o: paralleldecisiontree.cpp
	$(CC) $< -std=c++0x -o $@ testgame.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

decisiontree.o: decisiontree.cpp
	$(CC) $< -std=c++0x -o $@ testgame.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

deviceboard.o: deviceboard.cpp
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

board.o: board.cpp
	$(CC) $< -std=c++0x -o $@ testgame.o -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
