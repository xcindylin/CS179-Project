#include "exampleplayer.h"

ExamplePlayer::ExamplePlayer(Side side) {
    this->side = side;
    otherSide = side == BLACK ? WHITE : BLACK;
    board = new Board();
}

/*
 * Destructor for the player.
 */
ExamplePlayer::~ExamplePlayer() {
}

Move *ExamplePlayer::doMove(Move *opponentsMove) {
    board->doMove(opponentsMove, otherSide);
    vector<Move> moves = board->getMoves(side);
    if (moves.size() == 0) {
        return NULL;
    }
    return &(moves[0]);
}
    
