#include "player.h"

Player::Player(Side side) {
    this->side = side;
    otherSide = side == BLACK ? WHITE : BLACK;
    board = new Board();

}

Player::~Player() {
}

Move *Player::doMove(Move *opponentsMove) {
    board->doMove(opponentsMove, otherSide);
    if (!board->hasMoves(side)) {
        return NULL;
    }
    DecisionTree *tree = new DecisionTree(board, side);
    Move *moveToMake = tree->findBestMove(4);
    board->doMove(moveToMake, side);
    return moveToMake;
}
