#ifndef __NODE_H__
#define __NODE_H__

/* from http://stackoverflow.com/questions/6978643/cuda-and-classes */
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <cmath>
#include <vector>
#include "common.h"
#include "board.h"

class Node {

private:
	Board *board;
	Move *move;
	Side side; // the side that made the move leading to this node
	Side maximizer; // our side - maximizing side
	int score;
	int alpha;
	int beta;
	Node *parent;

public:
	CUDA_CALLABLE_MEMBER Node(Move *move, Side side, Side maximizer, Board *board);
	CUDA_CALLABLE_MEMBER ~Node();

	CUDA_CALLABLE_MEMBER Board *getBoard();
	CUDA_CALLABLE_MEMBER Move *getMove();
	CUDA_CALLABLE_MEMBER Side getSide();
	CUDA_CALLABLE_MEMBER Node *getParent();
	CUDA_CALLABLE_MEMBER void setParent(Node *node);
	CUDA_CALLABLE_MEMBER int getScore();
	CUDA_CALLABLE_MEMBER int getAlpha();
	CUDA_CALLABLE_MEMBER int getBeta();
	CUDA_CALLABLE_MEMBER void setAlpha(int alpha);
	CUDA_CALLABLE_MEMBER void setBeta(int beta);

};

#endif
