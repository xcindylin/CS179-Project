#ifndef __NODE_H__
#define __NODE_H__

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
	double score;
	double alpha;
	double beta;
	Node *parent;
	vector<Node *> children;

public:
	Node(Move *move, Side side, Side maximizer, Board *board);
	~Node();

	Board *getBoard();
	Move *getMove();
	Side getSide();
	Node *getParent();
	void setParent(Node *node);
	vector<Node *> getChildren();
	double getScore();
	double getAlpha();
	double getBeta();
	void setAlpha(double alpha);
	void setBeta(double beta);

	void addChild(Node *node);

};

#endif