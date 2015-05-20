#ifndef __PLAYER_H__
#define __PLAYER_H__

#include <iostream>
#include "common.h"
#include "board.h"
#include "decisiontree.h"
using namespace std;

class Player {

private:
	Side side;
	Side otherSide;
	Board *board;

public:
    Player(Side side);
    ~Player();
    
    Move *doMove(Move *opponentsMove);
};

#endif
