#include "node.h"

Node::Node(Move *move, Side side, Side maximizer, Board *board) {
	this->move = move;
	this->side = side;
	this->maximizer = maximizer;
	this->board = board;
	this->score = board->getScore(maximizer);
	this->alpha = -INFINITY;
	this->beta = INFINITY;
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

int Node::getScore() {
    return score;
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
