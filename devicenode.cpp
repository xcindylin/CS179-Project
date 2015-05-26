#include "devicenode.h"

DeviceNode::DeviceNode(Move *move, Side side, Side maximizer, DeviceBoard *board) {
	this->move = move;
	this->side = side;
	this->maximizer = maximizer;
	this->board = board;
	this->score = board->getScore(maximizer);
	this->alpha = -INFINITY;
	this->beta = INFINITY;
}

DeviceNode::~DeviceNode() {
	// free some stuff
	delete board;
}

DeviceBoard *DeviceNode::getBoard() {
	return board;
}

Move *DeviceNode::getMove() {
	return move;
}

Side DeviceNode::getSide() {
    return side;
}

DeviceNode *DeviceNode::getParent() {
	return parent;
}

void DeviceNode::setParent(DeviceNode *node) {
	parent = node;
}

int DeviceNode::getScore() {
    return score;
}

int DeviceNode::getAlpha() {
    return alpha;
}

int DeviceNode::getBeta() {
    return beta;
}

void DeviceNode::setAlpha(int alpha) {
    this->alpha = alpha;
}

void DeviceNode::setBeta(int beta) {
    this->beta = beta;
}
