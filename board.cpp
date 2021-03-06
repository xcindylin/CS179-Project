#include "board.h"

/*
 * Make a BOARD_SIZExBOARD_SIZE othello board and initialize it to the standard setup.
 */
Board::Board() {
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        taken[i] = 0;
        black[i] = 0;
    }
    // taken.set(7 + BOARD_SIZE * 7);
    taken[(BOARD_SIZE/2 - 1) + BOARD_SIZE * (BOARD_SIZE/2 - 1)] = 1;
    // taken.set(7 + BOARD_SIZE * 8);
    taken[(BOARD_SIZE/2 - 1) + BOARD_SIZE * (BOARD_SIZE/2)] = 1;
    // taken.set(8 + BOARD_SIZE * 7);
    taken[(BOARD_SIZE/2) + BOARD_SIZE * (BOARD_SIZE/2 - 1)] = 1;
    // taken.set(8 + BOARD_SIZE * 8);
    taken[(BOARD_SIZE/2) + BOARD_SIZE * (BOARD_SIZE/2)] = 1;
    // black.set(8 + BOARD_SIZE * 7);
    black[(BOARD_SIZE/2) + BOARD_SIZE * (BOARD_SIZE/2 - 1)] = 1;
    // black.set(7 + BOARD_SIZE * 8);
    black[(BOARD_SIZE/2 - 1) + BOARD_SIZE * (BOARD_SIZE/2)] = 1;
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
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        newBoard->black[i] = black[i];
        newBoard->taken[i] = taken[i];
    }
    return newBoard;
}

bool Board::occupied(int x, int y) {
    return taken[x + BOARD_SIZE*y];
}

bool Board::get(Side side, int x, int y) {
    return occupied(x, y) && (black[x + BOARD_SIZE*y] == boolToInt(side == BLACK));
}

void Board::set(Side side, int x, int y) {
    taken[x + BOARD_SIZE*y] = 1;
    // if side is black, side == BLACK will evaluate to 1
    black[x + BOARD_SIZE*y] = boolToInt(side == BLACK);
}

bool Board::onBoard(int x, int y) {
    return(0 <= x && x < BOARD_SIZE && 0 <= y && y < BOARD_SIZE);
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
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            Move move(i,j);
            if (checkMove(&move, side)) return true;
        }
    }
    return false;
}

// Returns a list of possible moves for the specified side
vector<Move> Board::getMoves(Side side) {
    vector<Move> movesList;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
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
    int count = 0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        count += black[i];
    }
    return count;
}

/*
 * Current count of white stones.
 */
int Board::countWhite() {
    int count = 0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        count += taken[i];
        count -= black[i];
    }
    return count;
}

/* 
 * Return the score of this board state for the maximizer.
 */
int Board::getScore(Side maximizer) {
    Side minimizer = maximizer == BLACK ? WHITE : BLACK;
    float score;

    if (maximizer == BLACK) {
        score = countBlack() - countWhite();
    } else {
        score = countWhite() - countBlack();
    }

    // update score by adding a positive weight if the maximizer has occupied a
    // corner or a negative weight if the minimizer has occupied a corner
    bool maxULCorner = get(maximizer, 0, 0);
    bool maxURCorner = get(maximizer, BOARD_SIZE-1, 0);
    bool maxLLCorner = get(maximizer, 0, BOARD_SIZE-1);
    bool maxLRCorner = get(maximizer, BOARD_SIZE-1, BOARD_SIZE-1);

    bool minULCorner = get(minimizer, 0, 0);
    bool minURCorner = get(minimizer, BOARD_SIZE-1, 0);
    bool minLLCorner = get(minimizer, 0, BOARD_SIZE-1);
    bool minLRCorner = get(minimizer, BOARD_SIZE-1, BOARD_SIZE-1);

    if (maxULCorner || maxURCorner || maxLLCorner || maxLRCorner) {
        score += CORNER_WEIGHT * (boolToInt(maxULCorner) + boolToInt(maxURCorner) 
                                + boolToInt(maxLLCorner) + boolToInt(maxLRCorner));
    }
    if (minULCorner || minURCorner || minLLCorner || minLRCorner) {
        score -= CORNER_WEIGHT * (boolToInt(minULCorner) + boolToInt(minURCorner) 
                                + boolToInt(minLLCorner) + boolToInt(minLRCorner));
    }

    // update score using a negative weight for positions in the diagonal that are
    // next to unoccupied corners
    bool maxULDiagonal = get(maximizer, 1, 1);
    bool maxURDiagonal = get(maximizer, BOARD_SIZE-2, 1);
    bool maxLLDiagonal = get(maximizer, 1, BOARD_SIZE-2);
    bool maxLRDiagonal = get(maximizer, BOARD_SIZE-2, BOARD_SIZE-2);

    if (maxULDiagonal && !minULCorner) {
        score += DIAGONAL_WEIGHT;
    }
    if (maxURDiagonal && !minURCorner) {
        score += DIAGONAL_WEIGHT;
    }
    if (maxLLDiagonal && !minLLCorner) {
        score += DIAGONAL_WEIGHT;
    }
    if (maxLRDiagonal && !minLRCorner) {
        score += DIAGONAL_WEIGHT;
    }

    // update score using a positive weight for occupied edge positions (edge
    // positions do not include the corners)
    for (int x = 0; x < BOARD_SIZE; x += BOARD_SIZE-1) {
        for (int y = 1; y < BOARD_SIZE-1; y++) {
            score += EDGE_WEIGHT * boolToInt(get(maximizer, x, y));
        }
    }
    for (int y = 0; y < BOARD_SIZE; y += BOARD_SIZE-1) {
        for (int x = 1; x < BOARD_SIZE-1; x++) {
            score += EDGE_WEIGHT * boolToInt(get(maximizer, x, y));
        }
    }

    score += MOVES_WEIGHT * getMovesScore(maximizer);
    score += FRONTIER_WEIGHT * getFrontierScore(maximizer);

    return score;
 }

// new functions

 int Board::boolToInt(bool b) {
    return b ? 1 : 0;
 }

int Board::getMovesScore(Side maximizer) {
    Side minimizer = maximizer == BLACK ? WHITE : BLACK;
    vector<Move> moves = getMoves(maximizer);
    vector<Move> opponentMoves = getMoves(minimizer);
    return moves.size()-opponentMoves.size();
}

int Board::getFrontierScore(Side maximizer) {
    int score = 0;
    bool frontier = false;

    for (int x = 1; x < BOARD_SIZE-1; x++) {
        for (int y = 1; y < BOARD_SIZE-1; y++) {
            // check to see if position on the board is occupied by
            // either the maximizer or minimizer
            if (occupied(x, y)) {
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        // continue since it's the current position
                        // being checked
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        // set flag since we have found an unoccupied
                        // position surrounding the current position
                        if (!occupied(x+dx, y+dy)) {
                            frontier = true;
                        }
                    }
                }
                if (frontier) {
                    if (get(maximizer, x, y)) {
                        // add to the score if maximizer is in the frontier
                        score--;
                    } else {
                        // subtract from the score if the minimizer is in
                        // the frontier
                        score++;
                    }
                }
            }
            frontier = false;
        }
    }
    return score;
}
