#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common.h"
#include "board.h"
#include "exampleplayer.h"

using namespace std;

Move *getMove(Side turn, Move *lastMove) {
	if (turn == BLACK) {
		ExamplePlayer *p = new ExamplePlayer(BLACK);
		return p->doMove(lastMove);
	}
	else {
		ExamplePlayer *p = new ExamplePlayer(WHITE);	
		return p->doMove(lastMove);	
	}
}

int main() {
	Board *board = new Board();
	// ExamplePlayer player1 = new ExamplePlayer(BLACK);
	// ExamplePlayer player2 = new ExamplePlayer(WHITE);

    Side turn = BLACK;
    Move *m = NULL;

    while (!board->isDone()) {
    	// get the current player's move
    	m = getMove(turn, m);

    	if (!board->checkMove(m, turn)) {
    		cout << "Illegal move made" << endl;
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
