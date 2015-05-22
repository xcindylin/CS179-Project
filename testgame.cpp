#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.h"
#include "board.h"
#include "exampleplayer.h"
#include "player.h"

using namespace std;

int main() {
	Board *board = new Board();
	ExamplePlayer *player1 = new ExamplePlayer(BLACK);
	Player *player2 = new Player(WHITE);

    Side turn = BLACK;
    Move *m = NULL;

    while (!board->isDone()) {
    	// get the current player's move
        if (turn == BLACK) {
            m = player1->doMove(m);
        }
        else {  
            m = player2->doMove(m);   
        }

    	if (!board->checkMove(m, turn)) {
    		cout << "Illegal move made: " << turn << " address: " << m << endl;
    	}

    	// make move once it is determiend to be legal
    	board->doMove(m, turn);

    	// switch players
    	if (turn == BLACK) {
    		turn = WHITE;
    	}
    	else {
    		turn = BLACK;
    	}
    }

    cout << "Game completed." << endl;
    cout << "Black score: " << board->countBlack() << endl;
    cout << "White score: " << board->countWhite() << endl;

    return 0;
}