#ifndef __TREE_CUDA_CUH__
#define __TREE_CUDA_CUH__

#include "common.h"
#include "node.h"
#include "board.h"

void cudaCallTreeKernel(Move *moves, Board *board, int *values, Side side, 
    Side maximizer, int alpha, int beta, int numMoves, int depth);

#endif