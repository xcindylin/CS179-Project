#include "node.h"

Node::Node(Move *move, Side side, Side maximizer, Board *board) {
	this->move = move;
	this->side = side;
	this->maximizer = maximizer;
	this->board = board;
	this->alpha = -99999999;
	this->beta = 99999999;
}

Node::~Node() {
	// free some stuff
	delete board;
}

Board *Node::getBoard() {
	return board;
}

Move *Node::getMove() {
	return move;
}

Side Node::getSide() {
    return side;
}

Node *Node::getParent() {
	return parent;
}

void Node::setParent(Node *node) {
	parent = node;
}

int Node::getAlpha() {
    return alpha;
}

int Node::getBeta() {
    return beta;
}

void Node::setAlpha(int alpha) {
    this->alpha = alpha;
}

void Node::setBeta(int beta) {
    this->beta = beta;
}
