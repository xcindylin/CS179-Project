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
	int score;
	int alpha;
	int beta;
	Node *parent;

public:
	Node(Move *move, Side side, Side maximizer, Board *board);
	~Node();

	Board *getBoard();
	Move *getMove();
	Side getSide();
	Node *getParent();
	void setParent(Node *node);
	int getScore();
	int getAlpha();
	int getBeta();
	void setAlpha(int alpha);
	void setBeta(int beta);

};

#endif
