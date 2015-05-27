#include "devicenode.h"

CUDA_CALLABLE_MEMBER
DeviceNode::DeviceNode(Move *move, Side side, Side maximizer, DeviceBoard *board) {
	this->move = move;
	this->side = side;
	this->maximizer = maximizer;
	this->board = board;
	this->score = board->getScore(maximizer);
	this->alpha = -INFINITY;
	this->beta = INFINITY;
}

CUDA_CALLABLE_MEMBER
DeviceNode::~DeviceNode() {
	// free some stuff
	delete board;
}

CUDA_CALLABLE_MEMBER
DeviceBoard *DeviceNode::getBoard() {
	return board;
}

CUDA_CALLABLE_MEMBER
Move *DeviceNode::getMove() {
	return move;
}

CUDA_CALLABLE_MEMBER
Side DeviceNode::getSide() {
    return side;
}

CUDA_CALLABLE_MEMBER
DeviceNode *DeviceNode::getParent() {
	return parent;
}

CUDA_CALLABLE_MEMBER
void DeviceNode::setParent(DeviceNode *node) {
	parent = node;
}

CUDA_CALLABLE_MEMBER
int DeviceNode::getScore() {
    return score;
}

CUDA_CALLABLE_MEMBER
int DeviceNode::getAlpha() {
    return alpha;
}

CUDA_CALLABLE_MEMBER
int DeviceNode::getBeta() {
    return beta;
}

CUDA_CALLABLE_MEMBER
void DeviceNode::setAlpha(int alpha) {
    this->alpha = alpha;
}

CUDA_CALLABLE_MEMBER
void DeviceNode::setBeta(int beta) {
    this->beta = beta;
}
