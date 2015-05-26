#include "devicenode.h"

__device__
DeviceNode::DeviceNode(Move *move, Side side, Side maximizer, DeviceBoard *board) {
	this->move = move;
	this->side = side;
	this->maximizer = maximizer;
	this->board = board;
	this->score = board->getScore(maximizer);
	this->alpha = -INFINITY;
	this->beta = INFINITY;
}

__device__
DeviceNode::~DeviceNode() {
	// free some stuff
	delete board;
}

__device__
DeviceBoard *DeviceNode::getBoard() {
	return board;
}

__device__
Move *DeviceNode::getMove() {
	return move;
}

__device__
Side DeviceNode::getSide() {
    return side;
}

__device__
DeviceNode *DeviceNode::getParent() {
	return parent;
}

__device__
void DeviceNode::setParent(DeviceNode *node) {
	parent = node;
}

__device__
int DeviceNode::getScore() {
    return score;
}

__device__
int DeviceNode::getAlpha() {
    return alpha;
}

__device__
int DeviceNode::getBeta() {
    return beta;
}

__device__
void DeviceNode::setAlpha(int alpha) {
    this->alpha = alpha;
}

__device__
void DeviceNode::setBeta(int beta) {
    this->beta = beta;
}
