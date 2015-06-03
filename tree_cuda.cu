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
bool checkMove(char *black, char *taken, int X, int Y, Side side) {

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
            if (checkMove(black, taken, i, j, side)) return true;
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
            if (checkMove(black, taken, i, j, side)) count++;
        }
    }
    return count;
}

/*
 * Modifies the board to reflect the specified move.
 */
__device__
void doMove(char *black, char *taken, int X, int Y, Side side) {
    // Ignore if move is invalid.
    if (!checkMove(black, taken, X, Y, side)) return;

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

__global__ 
void cudaSearchKernel(char *black, char *taken, int *alpha, int *beta, Side side, Side maximizer, int depth) {
    if (depth == 0) {
        *alpha = getScore(black, taken, maximizer);
        *beta = *alpha;
        return;
    }
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    while (index < BOARD_SIZE * BOARD_SIZE) {
        int x = index % BOARD_SIZE;
        int y = index / BOARD_SIZE;
        Side oppositeSide = side == BLACK ? WHITE : BLACK;
        if (checkMove(black, taken, x, y, oppositeSide)) {
            char *new_black;
            char *new_taken;

            new_black = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));
            new_taken = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));

            for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
                new_black[i] = black[i];
                new_taken[i] = taken[i];
            }
            doMove(new_black, new_taken, x, y, oppositeSide);

            int *new_alpha = (int *) malloc(sizeof(int));
            int *new_beta = (int *) malloc(sizeof(int));
            *new_alpha = *alpha;
            *new_beta = *beta;

            // search child
            cudaSearchKernel<<<1, 32>>>(new_black, new_taken, new_alpha, new_beta, oppositeSide, maximizer, depth - 1);
            cudaDeviceSynchronize();

            if (side == maximizer) {
                atomicMin(beta, *new_alpha);
            } else {
                atomicMax(alpha, *new_beta);
            }

            free(new_alpha);
            free(new_beta);
            free(new_black);
            free(new_taken);

            if (*alpha >= *beta) {
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
        doMove(new_black, new_taken, move->getX(), move->getY(), side);

        int *new_alpha = (int *) malloc(sizeof(int));
        int *new_beta = (int *) malloc(sizeof(int));
        *new_alpha = alpha;
        *new_beta = beta;

        cudaSearchKernel<<<1, 32>>>(new_black, new_taken, new_alpha, new_beta, side, maximizer, depth);
        cudaDeviceSynchronize();

        // update the values we care about - if the parent node is a maximizing node, 
        // it cares about the child alpha values
        if (side == maximizer) {
            values[blockIdx.x] = *new_beta;
        } else {
            values[blockIdx.x] = *new_alpha;
        }

        free(new_alpha);
        free(new_beta);
        free(new_black);
        free(new_taken);
    }
}

void cudaCallTreeKernel(Move *moves, char *black, char *taken, int *values, Side side, 
    Side maximizer, int alpha, int beta, int numMoves, int depth) {

    cudaTreeKernel<<<numMoves, 32>>>(moves, black, taken, values, side, 
       maximizer, alpha, beta, depth);
}
