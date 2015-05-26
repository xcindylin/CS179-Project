#ifndef __TREE_CUDA_CUH__
#define __TREE_CUDA_CUH__

#include "common.h"
#include "devicenode.h"
#include "deviceboard.h"

void cudaCallTreeKernel(Move *moves, DeviceBoard *board, int *values, Side side, 
    Side maximizer, int alpha, int beta, int numMoves, int depth);

#endif