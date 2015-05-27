#ifndef __DEVICEBOARD_H__
#define __DEVICEBOARD_H__

/* from http://stackoverflow.com/questions/6978643/cuda-and-classes */
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#include <thrust/device_vector.h>
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include "common.h"
#include <vector>

#define BOARD_SIZE 16

#define CORNER_WEIGHT 50
#define MOVES_WEIGHT 2 // # moves cur_player - # moves opp_player
#define DIAGONAL_WEIGHT -10
#define EDGE_WEIGHT 5
#define FRONTIER_WEIGHT 2

using namespace std;

class DeviceBoard {
   
private:
    CUDA_CALLABLE_MEMBER bool occupied(int x, int y);
    CUDA_CALLABLE_MEMBER bool get(Side side, int x, int y);
    CUDA_CALLABLE_MEMBER void set(Side side, int x, int y);
    CUDA_CALLABLE_MEMBER bool onBoard(int x, int y);
      
public:
    CUDA_CALLABLE_MEMBER DeviceBoard();
    CUDA_CALLABLE_MEMBER ~DeviceBoard();
    char black[BOARD_SIZE * BOARD_SIZE];
    char taken[BOARD_SIZE * BOARD_SIZE];
    CUDA_CALLABLE_MEMBER DeviceBoard *copy();
        
    CUDA_CALLABLE_MEMBER bool isDone();
    CUDA_CALLABLE_MEMBER bool hasMoves(Side side);
    CUDA_CALLABLE_MEMBER bool checkMove(Move *m, Side side);
    // __device__ thrust::device_vector<Move> getMoves(Side side);
    CUDA_CALLABLE_MEMBER int countMoves(Side side);
    CUDA_CALLABLE_MEMBER void doMove(Move *m, Side side);
    CUDA_CALLABLE_MEMBER int count(Side side);
    CUDA_CALLABLE_MEMBER int countBlack();
    CUDA_CALLABLE_MEMBER int countWhite();
    CUDA_CALLABLE_MEMBER int getScore(Side maximizer);

    // new functions
    CUDA_CALLABLE_MEMBER int boolToInt(bool b);
    CUDA_CALLABLE_MEMBER int getMovesScore(Side maximizer);
    CUDA_CALLABLE_MEMBER int getFrontierScore(Side maximizer);
};

#endif
