#ifndef __DEVICENODE_H__
#define __DEVICENODE_H__

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
	__device__ DeviceNode(Move *move, Side side, Side maximizer, DeviceBoard *board);
	__device__ ~DeviceNode();

	__device__ DeviceBoard *getBoard();
	__device__ Move *getMove();
	__device__ Side getSide();
	__device__ DeviceNode *getParent();
	__device__ void setParent(DeviceNode *node);
	__device__ int getScore();
	__device__ int getAlpha();
	__device__ int getBeta();
	__device__ void setAlpha(int alpha);
	__device__ void setBeta(int beta);

};

#endif
