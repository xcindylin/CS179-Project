#ifndef __GPUPLAYER_H__
#define __GPUPLAYER_H__

#include <iostream>
#include "common.h"
#include "board.h"
#include "decisiontree.h"
using namespace std;

class GPUPlayer {

private:
	Side side;
	Side otherSide;
	Board *board;

public:
    GPUPlayer(Side side);
    ~GPUPlayer();
    
    Move *doMove(Move *opponentsMove);
};

#endif