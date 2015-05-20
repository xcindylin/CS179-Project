#include "board.h"

/*
 * Make a 16x16 othello board and initialize it to the standard setup.
 */
Board::Board() {
    taken.set(7 + 16 * 7);
    taken.set(7 + 16 * 8);
    taken.set(8 + 16 * 7);
    taken.set(8 + 16 * 8);
    black.set(8 + 16 * 7);
    black.set(7 + 16 * 8);
}

/*
 * Destructor for the board.
 */
Board::~Board() {
}

/*
 * Returns a copy of this board.
 */
Board *Board::copy() {
    Board *newBoard = new Board();
    newBoard->black = black;
    newBoard->taken = taken;
    return newBoard;
}

bool Board::occupied(int x, int y) {
    return taken[x + 16*y];
}

bool Board::get(Side side, int x, int y) {
    return occupied(x, y) && (black[x + 16*y] == (side == BLACK));
}

void Board::set(Side side, int x, int y) {
    taken.set(x + 16*y);
    black.set(x + 16*y, side == BLACK);
}

bool Board::onBoard(int x, int y) {
    return(0 <= x && x < 16 && 0 <= y && y < 16);
}

 
/*
 * Returns true if the game is finished; false otherwise. The game is finished 
 * if neither side has a legal move.
 */
bool Board::isDone() {
    return !(hasMoves(BLACK) || hasMoves(WHITE));
}

/*
 * Returns true if there are legal moves for the given side.
 */
bool Board::hasMoves(Side side) {
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            Move move(i, j);
            if (checkMove(&move, side)) return true;
        }
    }
    return false;
}

// Returns a list of possible moves for the specified side
vector<Move> Board::getMoves(Side side) {
    vector<Move> movesList;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            Move move(i, j);
            if (checkMove(&move, side)) movesList.push_back(move);
        }
    }
    return movesList;
}

/*
 * Returns true if a move is legal for the given side; false otherwise.
 */
bool Board::checkMove(Move *m, Side side) {
    // Passing is only legal if you have no moves.
    if (m == NULL) return !hasMoves(side);

    int X = m->getX();
    int Y = m->getY();

    // Make sure the square hasn't already been taken.
    if (occupied(X, Y)) return false;

    Side other = (side == BLACK) ? WHITE : BLACK;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dy == 0 && dx == 0) continue;

            // Is there a capture in that direction?
            int x = X + dx;
            int y = Y + dy;
            if (onBoard(x, y) && get(other, x, y)) {
                do {
                    x += dx;
                    y += dy;
                } while (onBoard(x, y) && get(other, x, y));

                if (onBoard(x, y) && get(side, x, y)) return true;
            }
        }
    }
    return false;
}

/*
 * Modifies the board to reflect the specified move.
 */
void Board::doMove(Move *m, Side side) {
    // A NULL move means pass.
    if (m == NULL) return;

    // Ignore if move is invalid.
    if (!checkMove(m, side)) return;

    int X = m->getX();
    int Y = m->getY();
    Side other = (side == BLACK) ? WHITE : BLACK;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dy == 0 && dx == 0) continue;

            int x = X;
            int y = Y;
            do {
                x += dx;
                y += dy;
            } while (onBoard(x, y) && get(other, x, y));

            if (onBoard(x, y) && get(side, x, y)) {
                x = X;
                y = Y;
                x += dx;
                y += dy;
                while (onBoard(x, y) && get(other, x, y)) {
                    set(side, x, y);
                    x += dx;
                    y += dy;
                }
            }
        }
    }
    set(side, X, Y);
}

/*
 * Current count of given side's stones.
 */
int Board::count(Side side) {
    return (side == BLACK) ? countBlack() : countWhite();
}

/*
 * Current count of black stones.
 */
int Board::countBlack() {
    return black.count();
}

/*
 * Current count of white stones.
 */
int Board::countWhite() {
    return taken.count() - black.count();
}

/* 
 * Return the score of this board state for the maximizer.
 */
 float Board::getScore(Side maximizer) {
    // TODO
    float score;
    if (maximizer == BLACK) {
        score = countBlack() - countWhite();
    } else {
        score = countWhite() - countBlack();
    }
    return score;
 }
