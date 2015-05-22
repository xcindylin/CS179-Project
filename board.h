#ifndef __BOARD_H__
#define __BOARD_H__

#include <bitset>
#include "common.h"
#include <vector>

#define BOARD_SIZE 16

#define CORNER_WEIGHT 50
#define MOVES_WEIGHT 2 // # moves cur_player - # moves opp_player
#define DIAGONAL_WEIGHT -10
#define EDGE_WEIGHT 5
#define FRONTIER_WEIGHT 2

using namespace std;

class Board {
   
private:
    bitset<BOARD_SIZE * BOARD_SIZE> black;
    bitset<BOARD_SIZE * BOARD_SIZE> taken;    
       
    bool occupied(int x, int y);
    bool get(Side side, int x, int y);
    void set(Side side, int x, int y);
    bool onBoard(int x, int y);
      
public:
    Board();
    ~Board();
    Board *copy();
        
    bool isDone();
    bool hasMoves(Side side);
    bool checkMove(Move *m, Side side);
    vector<Move> getMoves(Side side);
    void doMove(Move *m, Side side);
    int count(Side side);
    int countBlack();
    int countWhite();
    float getScore(Side maximizer);

    // new functions
    int boolToInt(bool b);
    int getMovesScore(Side maximizer);
    int getFrontierScore(Side maximizer);
};

#endif
