#ifndef __DECISIONTREE_H__
#define __DECISIONTREE_H__

#include <iostream>
#include "common.h"
#include "board.h"
#include "node.h"
using namespace std;

class DecisionTree { 

private:
	Node *root;

protected:
	Side maximizer;

public:
	DecisionTree(Board *board, Side side);
	~DecisionTree();

	Move *findBestMove(int depth);
	void search(Node *startingNode, int depth);

};

#endif
