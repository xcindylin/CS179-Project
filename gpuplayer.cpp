#include "gpuplayer.h"

GPUPlayer::GPUPlayer(Side side) {
    this->side = side;
    otherSide = side == BLACK ? WHITE : BLACK;
    board = new Board();

}

GPUPlayer::~GPUPlayer() {
}

Move *Player::doMove(Move *opponentsMove) {
    board->doMove(opponentsMove, otherSide);
    if (!board->hasMoves(side)) {
        return NULL;
    }
    ParallelDecisionTree *tree = new DecisionTree(board, side);
    Move *moveToMake = tree->search(3);
    board->doMove(moveToMake, side);
    return moveToMake;
}
