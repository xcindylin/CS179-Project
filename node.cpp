#include "node.h"

Node::Node(Move *move, Side side, Side maximizer, Board *board) {
	this->move = move;
	this->side = side;
	this->maximizer = maximizer;
	this->board = board;
	this->score = board.getScore(maximizer);
	this->alpha = -INFINITY;
	this->beta = INFINITY;
}

Node::~Node() {
	// free some stuff
}

Board *Node::getBoard() {
	return board;
}

Move *Node::getMove() {
	return move;
}

Node *Node::getParent() {
	return parent;
}

void Node::setParent(Node *node) {
	parent = node;
}

vector<Node *> Node::getChildren() {
	return children;
}

double Node::getScore() {
    return score;
}

double Node::getAlpha() {
    return alpha;
}

double Node::getBeta() {
    return beta;
}

void Node::setAlpha(double alpha) {
    this->alpha = alpha;
}

void Node::setBeta(double beta) {
    this->beta = beta;
}

void Node::addChild(Node *node) {
	children.push_back(node);
	node.setParent(this);
}