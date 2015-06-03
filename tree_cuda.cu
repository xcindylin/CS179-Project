#include <cstdio>
#include "tree_cuda.cuh"

 __device__
 int boolToInt(bool b) {
    return b ? 1 : 0;
 }

__device__
bool occupied(char *taken, int x, int y) {
    return taken[x + BOARD_SIZE*y];
}

__device__
bool get(char *black, char *taken, Side side, int x, int y) {
    return occupied(taken, x, y) && (black[x + BOARD_SIZE*y] == boolToInt(side == BLACK));
}

__device__
void set(char *black, char *taken, Side side, int x, int y) {
    taken[x + BOARD_SIZE*y] = 1;
    // if side is black, side == BLACK will evaluate to 1
    black[x + BOARD_SIZE*y] = boolToInt(side == BLACK);
}

__device__
bool onBoard(int x, int y) {
    return(0 <= x && x < BOARD_SIZE && 0 <= y && y < BOARD_SIZE);
}

/*
 * Returns true if a move is legal for the given side; false otherwise.
 */
__device__
bool checkMove(char *black, char *taken, Move *m, Side side) {
    // Passing is only legal if you have no moves.
    if (m == NULL) return true;
    // if (m == NULL) return !hasMoves(black, taken, side);

    int X = m->getX();
    int Y = m->getY();

    // Make sure the square hasn't already been taken.
    if (occupied(taken, X, Y)) return false;

    Side other = (side == BLACK) ? WHITE : BLACK;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dy == 0 && dx == 0) continue;

            // Is there a capture in that direction?
            int x = X + dx;
            int y = Y + dy;
            if (onBoard(x, y) && get(black, taken, other, x, y)) {
                do {
                    x += dx;
                    y += dy;
                } while (onBoard(x, y) && get(black, taken, other, x, y));

                if (onBoard(x, y) && get(black, taken, side, x, y)) return true;
            }
        }
    }
    return false;
}

/*
 * Returns true if there are legal moves for the given side.
 */
__device__
bool hasMoves(char *black, char *taken, Side side) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            Move move(i,j);
            if (checkMove(black, taken, &move, side)) return true;
        }
    }
    return false;
}

/*
 * Returns true if the game is finished; false otherwise. The game is finished 
 * if neither side has a legal move.
 */
__device__
bool isDone(char *black, char *taken) {
    return !(hasMoves(black, taken, BLACK) || hasMoves(black, taken, WHITE));
}

__device__
int countMoves(char *black, char *taken, Side side) {
    // find moves on the GPU
    int count = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            Move move(i, j);
            if (checkMove(black, taken, &move, side)) count++;
        }
    }
    return count;
}

/*
 * Modifies the board to reflect the specified move.
 */
__device__
void doMove(char *black, char *taken, Move *m, Side side) {
    // A NULL move means pass.
    if (m == NULL) return;

    // Ignore if move is invalid.
    if (!checkMove(black, taken, m, side)) return;

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
            } while (onBoard(x, y) && get(black, taken, other, x, y));

            if (onBoard(x, y) && get(black, taken, side, x, y)) {
                x = X;
                y = Y;
                x += dx;
                y += dy;
                while (onBoard(x, y) && get(black, taken, other, x, y)) {
                    set(black, taken, side, x, y);
                    x += dx;
                    y += dy;
                }
            }
        }
    }
    set(black, taken, side, X, Y);
}

/*
 * Current count of black stones.
 */
__device__
int countBlack(char *black, char *taken) {
    int count = 0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        count += black[i];
    }
    return count;
}

/*
 * Current count of white stones.
 */
__device__
int countWhite(char *black, char *taken) {
    int count = 0;
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        count += taken[i];
        count -= black[i];
    }
    return count;
}

/*
 * Current count of given side's stones.
 */
__device__
int count(char *black, char *taken, Side side) {
    return (side == BLACK) ? countBlack(black, taken) : countWhite(black, taken);
}

__device__
int getMovesScore(char *black, char *taken, Side maximizer) {
    Side minimizer = maximizer == BLACK ? WHITE : BLACK;
    int maximizerCount = countMoves(black, taken, maximizer);
    int minimizerCount = countMoves(black, taken, minimizer);
    return maximizerCount - minimizerCount;
}

__device__
int getFrontierScore(char *black, char *taken, Side maximizer) {
    int score = 0;
    bool frontier = false;

    for (int x = 1; x < BOARD_SIZE-1; x++) {
        for (int y = 1; y < BOARD_SIZE-1; y++) {
            // check to see if position on the board is occupied by
            // either the maximizer or minimizer
            if (occupied(taken, x, y)) {
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        // continue since it's the current position
                        // being checked
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        // set flag since we have found an unoccupied
                        // position surrounding the current position
                        if (!occupied(taken, x+dx, y+dy)) {
                            frontier = true;
                        }
                    }
                }
                if (frontier) {
                    if (get(black, taken, maximizer, x, y)) {
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

/* 
 * Return the score of this board state for the maximizer.
 */
__device__
int getScore(char *black, char *taken, Side maximizer) {
    Side minimizer = maximizer == BLACK ? WHITE : BLACK;
    float score;

    if (maximizer == BLACK) {
        score = countBlack(black, taken) - countWhite(black, taken);
    } else {
        score = countWhite(black, taken) - countBlack(black, taken);
    }

    // update score by adding a positive weight if the maximizer has occupied a
    // corner or a negative weight if the minimizer has occupied a corner
    bool maxULCorner = get(black, taken, maximizer, 0, 0);
    bool maxURCorner = get(black, taken, maximizer, BOARD_SIZE-1, 0);
    bool maxLLCorner = get(black, taken, maximizer, 0, BOARD_SIZE-1);
    bool maxLRCorner = get(black, taken, maximizer, BOARD_SIZE-1, BOARD_SIZE-1);

    bool minULCorner = get(black, taken, minimizer, 0, 0);
    bool minURCorner = get(black, taken, minimizer, BOARD_SIZE-1, 0);
    bool minLLCorner = get(black, taken, minimizer, 0, BOARD_SIZE-1);
    bool minLRCorner = get(black, taken, minimizer, BOARD_SIZE-1, BOARD_SIZE-1);

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
    bool maxULDiagonal = get(black, taken, maximizer, 1, 1);
    bool maxURDiagonal = get(black, taken, maximizer, BOARD_SIZE-2, 1);
    bool maxLLDiagonal = get(black, taken, maximizer, 1, BOARD_SIZE-2);
    bool maxLRDiagonal = get(black, taken, maximizer, BOARD_SIZE-2, BOARD_SIZE-2);

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
            score += EDGE_WEIGHT * boolToInt(get(black, taken, maximizer, x, y));
        }
    }
    for (int y = 0; y < BOARD_SIZE; y += BOARD_SIZE-1) {
        for (int x = 1; x < BOARD_SIZE-1; x++) {
            score += EDGE_WEIGHT * boolToInt(get(black, taken, maximizer, x, y));
        }
    }

    score += MOVES_WEIGHT * getMovesScore(black, taken, maximizer);
    score += FRONTIER_WEIGHT * getFrontierScore(black, taken, maximizer);

    return score;
 }


// __global__
// void cudaCountMovesKernel(DeviceBoard *board, Side side, int *score) {
//     unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

//     while (index < BOARD_SIZE * BOARD_SIZE) {
//         int x = index % BOARD_SIZE;
//         int y = index / BOARD_SIZE;
//         Move *move = new Move(x, y);
//         if (board->checkMove(move, side)) {
//             atomicAdd(score, 1);
//         }
//     }
// }

// __global__
// void cudaGetFrontierScore(DeviceBoard *board, Side maximizer, int *score) {
//     bool frontier = false;

//     unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

//     while (index < BOARD_SIZE * BOARD_SIZE) {
//         // first row, second column to column before the last column
//         if ( (index > 0) && (index < BOARD_SIZE-1) ) {
//             continue;
//         }
//         // last row, second column to column before the last column
//         if ( (index > (BOARD_SIZE-1)*BOARD_SIZE) && (index < BOARD_SIZE*BOARD_SIZE-1) ) {
//             continue;
//         }
//         // first column of board
//         if (index % BOARD_SIZE == 0) {
//             continue;
//         }
//        // last column of board
//         if ( (index+1) & BOARD_SIZE == 0) {
//             continue;
//         }

//         int x = index % BOARD_SIZE;
//         int y = index / BOARD_SIZE;

//         if (board->occupied(x, y)) {
//             for (int dx = -1; dx <= 1; dx++) {
//                 for (int dy = -1; dy <= 1; dy++) {
//                     // continue since it's the current position
//                     // being checked
//                     if (dx == 0 && dy == 0) {
//                         continue;
//                     }
//                     // set flag since we have found an unoccupied
//                     // position surrounding the current position
//                     if (!board->occupied(x+dx, y+dy)) {
//                         frontier = true;
//                     }
//                 }
//             }
//             if (frontier) {
//                 if (board->get(maximizer, x, y)) {
//                     // add to the score if maximizer is in the frontier
//                     atomicAdd(score, -1);
//                 } else {
//                     // subtract from the score if the minimizer is in
//                     // the frontier
//                     atomicAdd(score, 1);
//                 }
//             }
//         }
//         frontier = false;
//     }
// }

__device__ 
void cudaSearch(DeviceNode *node, Side side, Side maximizer, int depth) {
    DeviceBoard *board = node->getBoard();
    Side oppositeSide = side == BLACK ? WHITE : BLACK;
    
    if (depth == 0) {
        node->setAlpha(board->getScore(maximizer));
        node->setBeta(board->getScore(maximizer));
        return;
    }

    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            Move *move = new Move(i, j);
            if (board->checkMove(move, oppositeSide)) {
                char *black;
                char *taken;

                black = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));
                taken = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));

                for (int k = 0; k < BOARD_SIZE * BOARD_SIZE; k++) {
                    black[k] = board->black[k];
                    taken[k] = board->taken[k];
                }
                DeviceBoard *newBoard = new DeviceBoard(black, taken);
                newBoard->doMove(move, oppositeSide);
                DeviceNode *child = new DeviceNode(move, oppositeSide, maximizer, newBoard);

                // pass alpha and beta values down
                child->setAlpha(node->getAlpha());
                child->setBeta(node->getBeta());

                // search child
                cudaSearch(child, oppositeSide, maximizer, depth - 1);

                if (side == maximizer) {
                    node->setBeta(min(node->getBeta(), child->getAlpha()));
                } else {
                    node->setAlpha(max(node->getAlpha(), child->getBeta()));
                }

                delete child;
                if (node->getAlpha() >= node->getBeta()) {
                    return;
                }
            }
            delete move;
        }
    }
}

// __device__
// int cudaGetScore(DeviceBoard *board, Side maximizer) {
//     Side minimizer = maximizer == BLACK ? WHITE : BLACK;
//     int score;

//     if (maximizer == BLACK) {
//         score = board->countBlack() - board->countWhite();
//     } else {
//         score = board->countWhite() - board->countBlack();
//     }

//     // update score by adding a positive weight if the maximizer has occupied a
//     // corner or a negative weight if the minimizer has occupied a corner
//     bool maxULCorner = board->get(maximizer, 0, 0);
//     bool maxURCorner = board->get(maximizer, BOARD_SIZE-1, 0);
//     bool maxLLCorner = board->get(maximizer, 0, BOARD_SIZE-1);
//     bool maxLRCorner = board->get(maximizer, BOARD_SIZE-1, BOARD_SIZE-1);

//     bool minULCorner = board->get(minimizer, 0, 0);
//     bool minURCorner = board->get(minimizer, BOARD_SIZE-1, 0);
//     bool minLLCorner = board->get(minimizer, 0, BOARD_SIZE-1);
//     bool minLRCorner = board->get(minimizer, BOARD_SIZE-1, BOARD_SIZE-1);

//     if (maxULCorner || maxURCorner || maxLLCorner || maxLRCorner) {
//         score += CORNER_WEIGHT * (board->boolToInt(maxULCorner) + board->boolToInt(maxURCorner) 
//                                 + board->boolToInt(maxLLCorner) + board->boolToInt(maxLRCorner));
//     }
//     if (minULCorner || minURCorner || minLLCorner || minLRCorner) {
//         score -= CORNER_WEIGHT * (board->boolToInt(minULCorner) + board->boolToInt(minURCorner) 
//                                 + board->boolToInt(minLLCorner) + board->boolToInt(minLRCorner));
//     }

//     // update score using a negative weight for positions in the diagonal that are
//     // next to unoccupied corners
//     bool maxULDiagonal = board->get(maximizer, 1, 1);
//     bool maxURDiagonal = board->get(maximizer, BOARD_SIZE-2, 1);
//     bool maxLLDiagonal = board->get(maximizer, 1, BOARD_SIZE-2);
//     bool maxLRDiagonal = board->get(maximizer, BOARD_SIZE-2, BOARD_SIZE-2);

//     if (maxULDiagonal && !minULCorner) {
//         score += DIAGONAL_WEIGHT;
//     }
//     if (maxURDiagonal && !minURCorner) {
//         score += DIAGONAL_WEIGHT;
//     }
//     if (maxLLDiagonal && !minLLCorner) {
//         score += DIAGONAL_WEIGHT;
//     }
//     if (maxLRDiagonal && !minLRCorner) {
//         score += DIAGONAL_WEIGHT;
//     }

//     // update score using a positive weight for occupied edge positions (edge
//     // positions do not include the corners)
//     for (int x = 0; x < BOARD_SIZE; x += BOARD_SIZE-1) {
//         for (int y = 1; y < BOARD_SIZE-1; y++) {
//             score += EDGE_WEIGHT * board->boolToInt(board->get(maximizer, x, y));
//         }
//     }
//     for (int y = 0; y < BOARD_SIZE; y += BOARD_SIZE-1) {
//         for (int x = 1; x < BOARD_SIZE-1; x++) {
//             score += EDGE_WEIGHT * board->boolToInt(board->get(maximizer, x, y));
//         }
//     }

//     int *maximizerMovesScore;
//     int *minimizerMovesScore;
//     int *frontierScore;
//     maximizerMovesScore = (int *) malloc(sizeof(int));
//     minimizerMovesScore = (int *) malloc(sizeof(int));
//     frontierScore = (int *) malloc(sizeof(int));
//     *maximizerMovesScore = 0;
//     *minimizerMovesScore = 0;
//     *frontierScore = 0;

//     cudaCountMovesKernel<<<8, 64>>>(board, maximizer, maximizerMovesScore);
//     cudaCountMovesKernel<<<8, 64>>>(board, minimizer, minimizerMovesScore);
//     cudaGetFrontierScore<<<8, 64>>>(board, maximizer, frontierScore);

//     score += MOVES_WEIGHT * (*maximizerMovesScore - *minimizerMovesScore);
//     score += FRONTIER_WEIGHT * (*frontierScore);

//     return score;
//  }

__global__ 
void cudaSearchKernel(char *black, char *taken, int alpha, int beta, Side side, Side maximizer, int *value, int depth) {
    if (depth == 0) {
        *value = getScore(black, taken, maximizer);
        return;
    }
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    while (index < BOARD_SIZE * BOARD_SIZE) {
        int x = index % BOARD_SIZE;
        int y = index / BOARD_SIZE;
        Side oppositeSide = side == BLACK ? WHITE : BLACK;
        Move move(x, y);
        if (checkMove(black, taken, &move, oppositeSide)) {
            char *new_black;
            char *new_taken;

            new_black = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));
            new_taken = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));

            for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
                new_black[i] = black[i];
                new_taken[i] = taken[i];
            }
            doMove(new_black, new_taken, &move, oppositeSide);

            int *child_value = (int *) malloc(sizeof(int));
            *child_value = 0;

            // search child
            cudaSearchKernel<<<1, 32>>>(new_black, new_taken, alpha, beta, oppositeSide, maximizer, child_value, depth - 1);

            if (side == maximizer) {
                beta = min(beta, *child_value);
                *value = beta;
            } else {
                alpha = max(alpha, *child_value);
                *value = alpha;
            }

            free(child_value);
            free(new_black);
            free(new_taken);

            if (alpha >= beta) {
                return;
            }
        }
        index += blockDim.x * gridDim.x;
    }
}

__global__
void cudaTreeKernel(Move *moves, char *black, char *taken, int *values, Side side, 
    Side maximizer, int alpha, int beta, int depth) {
    // only one thread does high-level tasks
    if (threadIdx.x == 0) {
        // make one new node per block
        Move *move = new Move(moves[blockIdx.x].getX(), moves[blockIdx.x].getY());

        char *new_black;
        char *new_taken;

        new_black = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));
        new_taken = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));

        for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
            new_black[i] = black[i];
            new_taken[i] = taken[i];
        }
        doMove(new_black, new_taken, move, side);

        int *value = (int *) malloc(sizeof(int));
        *value = 0;

        cudaSearchKernel<<<1, 32>>>(new_black, new_taken, alpha, beta, side, maximizer, value, depth);

        // update the values we care about - if the parent node is a maximizing node, 
        // it cares about the child alpha values
        values[blockIdx.x] = *value;

        free(value);
        free(new_black);
        free(new_taken);
    }
}

void cudaCallTreeKernel(Move *moves, char *black, char *taken, int *values, Side side, 
    Side maximizer, int alpha, int beta, int numMoves, int depth) {

    cudaTreeKernel<<<numMoves, 32>>>(moves, black, taken, values, side, 
       maximizer, alpha, beta, depth);
}
