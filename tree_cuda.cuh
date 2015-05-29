#ifndef __TREE_CUDA_CUH__
#define __TREE_CUDA_CUH__

#include <cstdio>
#include "common.h"
#include "devicenode.h"
#include "deviceboard.h"

void cudaCallTreeKernel(Move *moves, char *black, char *taken, int *values, Side side, 
    Side maximizer, int alpha, int beta, int numMoves, int depth);

#endif