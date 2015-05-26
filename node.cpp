#include "node.h"

CUDA_CALLABLE_MEMBER
Node::Node(Move *move, Side side, Side maximizer, Board *board) {
	this->move = move;
	this->side = side;
	this->maximizer = maximizer;
	this->board = board;
	this->score = board->getScore(maximizer);
	this->alpha = -INFINITY;
	this->beta = INFINITY;
}

CUDA_CALLABLE_MEMBER
Node::~Node() {
	// free some stuff
	delete board;
}

CUDA_CALLABLE_MEMBER
Board *Node::getBoard() {
	return board;
}

CUDA_CALLABLE_MEMBER
Move *Node::getMove() {
	return move;
}

CUDA_CALLABLE_MEMBER
Side Node::getSide() {
    return side;
}

CUDA_CALLABLE_MEMBER
Node *Node::getParent() {
	return parent;
}

CUDA_CALLABLE_MEMBER
void Node::setParent(Node *node) {
	parent = node;
}

CUDA_CALLABLE_MEMBER
int Node::getScore() {
    return score;
}

CUDA_CALLABLE_MEMBER
int Node::getAlpha() {
    return alpha;
}

CUDA_CALLABLE_MEMBER
int Node::getBeta() {
    return beta;
}

CUDA_CALLABLE_MEMBER
void Node::setAlpha(int alpha) {
    this->alpha = alpha;
}

CUDA_CALLABLE_MEMBER
void Node::setBeta(int beta) {
    this->beta = beta;
}
