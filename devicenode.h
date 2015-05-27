#ifndef __DEVICENODE_H__
#define __DEVICENODE_H__

/* from http://stackoverflow.com/questions/6978643/cuda-and-classes */
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#include <thrust/device_vector.h>
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <cmath>
#include <vector>
#include "common.h"
#include "deviceboard.h"

class DeviceNode {

private:
	DeviceBoard *board;
	Move *move;
	Side side; // the side that made the move leading to this node
	Side maximizer; // our side - maximizing side
	int score;
	int alpha;
	int beta;
	DeviceNode *parent;

public:
	CUDA_CALLABLE_MEMBER DeviceNode(Move *move, Side side, Side maximizer, DeviceBoard *board);
	CUDA_CALLABLE_MEMBER ~DeviceNode();

	CUDA_CALLABLE_MEMBER DeviceBoard *getBoard();
	CUDA_CALLABLE_MEMBER Move *getMove();
	CUDA_CALLABLE_MEMBER Side getSide();
	CUDA_CALLABLE_MEMBER DeviceNode *getParent();
	CUDA_CALLABLE_MEMBER void setParent(DeviceNode *node);
	CUDA_CALLABLE_MEMBER int getScore();
	CUDA_CALLABLE_MEMBER int getAlpha();
	CUDA_CALLABLE_MEMBER int getBeta();
	CUDA_CALLABLE_MEMBER void setAlpha(int alpha);
	CUDA_CALLABLE_MEMBER void setBeta(int beta);

};

#endif
