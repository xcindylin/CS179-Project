#ifndef __PARALLELDECISIONTREE_H__
#define __PARALLELDECISIONTREE_H__

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#include "common.h"
#include "board.h"
#include "node.h"
#include "tree_cuda.cuh"

using namespace std;

class ParallelDecisionTree { 

private:
	Node *root;

protected:
	Side maximizer;

public:
    ParallelDecisionTree(Board *board, Side side);
	~ParallelDecisionTree();

	DeviceBoard *HostToDeviceBoard(Board *board);
	Move *search(Node *startingNode, int depth);
	Node *getRoot();
};

#endif
