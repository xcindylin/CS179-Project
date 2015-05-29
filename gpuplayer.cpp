#include "gpuplayer.h"

GPUPlayer::GPUPlayer(Side side) {
    this->side = side;
    otherSide = side == BLACK ? WHITE : BLACK;
    board = new Board();

}

GPUPlayer::~GPUPlayer() {
}

Move *GPUPlayer::doMove(Move *opponentsMove) {
    board->doMove(opponentsMove, otherSide);
    if (!board->hasMoves(side)) {
        return NULL;
    }
    ParallelDecisionTree *tree = new ParallelDecisionTree(board, side);
    Move *moveToMake = tree->search(tree->getRoot(), 2);
    board->doMove(moveToMake, side);
    return moveToMake;
}
