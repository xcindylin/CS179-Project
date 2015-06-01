#include <cstdio>
#include "tree_cuda.cuh"

__global__
void cudaCountMovesKernel(DeviceBoard *board, Side side, int *score) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    while (index < BOARD_SIZE * BOARD_SIZE) {
        int x = index % BOARD_SIZE;
        int y = index / BOARD_SIZE;
        Move *move = new Move(x, y);
        if (board->checkMove(move, side)) {
            atomicAdd(score, 1);
        }

        delete move;
        index += blockDim.x * gridDim.x;
    }
}

__global__
void cudaGetFrontierScore(DeviceBoard *board, Side maximizer, int *score) {
    bool frontier = false;

    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    while (index < BOARD_SIZE * BOARD_SIZE) {
        // first row, second column to column before the last column
        if ( (index > 0) && (index < BOARD_SIZE-1) ) {
            index += blockDim.x * gridDim.x;
            continue;
        }
        // last row, second column to column before the last column
        if ( (index > (BOARD_SIZE-1)*BOARD_SIZE) && (index < BOARD_SIZE*BOARD_SIZE-1) ) {
            index += blockDim.x * gridDim.x;
            continue;
        }
        // first column of board
        if (index % BOARD_SIZE == 0) {
            index += blockDim.x * gridDim.x;
            continue;
        }
       // last column of board
        if ( (index+1) % BOARD_SIZE == 0) {
            index += blockDim.x * gridDim.x;
            continue;
        }

        int x = index % BOARD_SIZE;
        int y = index / BOARD_SIZE;

        if (board->occupied(x, y)) {
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    // continue since it's the current position
                    // being checked
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    // set flag since we have found an unoccupied
                    // position surrounding the current position
                    if (!board->occupied(x+dx, y+dy)) {
                        frontier = true;
                    }
                }
            }
            if (frontier) {
                if (board->get(maximizer, x, y)) {
                    // add to the score if maximizer is in the frontier
                    atomicAdd(score, -1);
                } 
                else {
                    // subtract from the score if the minimizer is in
                    // the frontier
                    atomicAdd(score, 1);
                }
            }
        }

        frontier = false;
        index += blockDim.x * gridDim.x;
    }
}

__device__
int cudaGetScore(DeviceBoard *board, Side maximizer) {
    Side minimizer = maximizer == BLACK ? WHITE : BLACK;
    
    int *maximizerMovesScore;
    int *minimizerMovesScore;
    int *frontierScore;
    maximizerMovesScore = (int *) malloc(sizeof(int));
    minimizerMovesScore = (int *) malloc(sizeof(int));
    frontierScore = (int *) malloc(sizeof(int));
    *maximizerMovesScore = 0;
    *minimizerMovesScore = 0;
    *frontierScore = 0;

    cudaCountMovesKernel<<<2, 64>>>(board, maximizer, maximizerMovesScore);
    cudaCountMovesKernel<<<2, 64>>>(board, minimizer, minimizerMovesScore);
    cudaGetFrontierScore<<<2, 64>>>(board, maximizer, frontierScore);
    cudaDeviceSynchronize();

    int score;

    if (maximizer == BLACK) {
        score = board->countBlack() - board->countWhite();
    } else {
        score = board->countWhite() - board->countBlack();
    }

    // update score by adding a positive weight if the maximizer has occupied a
    // corner or a negative weight if the minimizer has occupied a corner
    bool maxULCorner = board->get(maximizer, 0, 0);
    bool maxURCorner = board->get(maximizer, BOARD_SIZE-1, 0);
    bool maxLLCorner = board->get(maximizer, 0, BOARD_SIZE-1);
    bool maxLRCorner = board->get(maximizer, BOARD_SIZE-1, BOARD_SIZE-1);

    bool minULCorner = board->get(minimizer, 0, 0);
    bool minURCorner = board->get(minimizer, BOARD_SIZE-1, 0);
    bool minLLCorner = board->get(minimizer, 0, BOARD_SIZE-1);
    bool minLRCorner = board->get(minimizer, BOARD_SIZE-1, BOARD_SIZE-1);

    if (maxULCorner || maxURCorner || maxLLCorner || maxLRCorner) {
        score += CORNER_WEIGHT * (board->boolToInt(maxULCorner) + board->boolToInt(maxURCorner) 
                                + board->boolToInt(maxLLCorner) + board->boolToInt(maxLRCorner));
    }
    if (minULCorner || minURCorner || minLLCorner || minLRCorner) {
        score -= CORNER_WEIGHT * (board->boolToInt(minULCorner) + board->boolToInt(minURCorner) 
                                + board->boolToInt(minLLCorner) + board->boolToInt(minLRCorner));
    }

    // update score using a negative weight for positions in the diagonal that are
    // next to unoccupied corners
    bool maxULDiagonal = board->get(maximizer, 1, 1);
    bool maxURDiagonal = board->get(maximizer, BOARD_SIZE-2, 1);
    bool maxLLDiagonal = board->get(maximizer, 1, BOARD_SIZE-2);
    bool maxLRDiagonal = board->get(maximizer, BOARD_SIZE-2, BOARD_SIZE-2);

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
            score += EDGE_WEIGHT * board->boolToInt(board->get(maximizer, x, y));
        }
    }
    for (int y = 0; y < BOARD_SIZE; y += BOARD_SIZE-1) {
        for (int x = 1; x < BOARD_SIZE-1; x++) {
            score += EDGE_WEIGHT * board->boolToInt(board->get(maximizer, x, y));
        }
    }

    score += MOVES_WEIGHT * (*maximizerMovesScore - *minimizerMovesScore);
    score += FRONTIER_WEIGHT * (*frontierScore);

    free(maximizerMovesScore);
    free(minimizerMovesScore);
    free(frontierScore);

    return score;
 }

__device__ 
void cudaSearch(DeviceNode *node, Side side, Side maximizer, int depth) {
    DeviceBoard *board = node->getBoard();
    Side oppositeSide = side == BLACK ? WHITE : BLACK;
    
    if (depth == 0) {
        int score = cudaGetScore(board, maximizer);
        node->setAlpha(score);
        node->setBeta(score);
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

                for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
                    black[i] = board->black[i];
                    taken[i] = board->taken[i];
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

        DeviceBoard *newBoard = new DeviceBoard(new_black, new_taken);
        newBoard->doMove(move, side);
        DeviceNode *node = new DeviceNode(move, side, maximizer, newBoard);

        // pass down alpha and beta
        node->setAlpha(alpha);
        node->setBeta(beta);

        cudaSearch(node, side, maximizer, depth);

        // update the values we care about - if the parent node is a maximizing node, 
        // it cares about the child alpha values
        if (side == maximizer) {
            values[blockIdx.x] = node->getBeta();
        } else {
            values[blockIdx.x] = node->getAlpha();
        }

        delete node;
        delete move;
    }
}

void cudaCallTreeKernel(Move *moves, char *black, char *taken, int *values, Side side, 
    Side maximizer, int alpha, int beta, int numMoves, int depth) {

    cudaTreeKernel<<<numMoves, 32>>>(moves, black, taken, values, side, 
       maximizer, alpha, beta, depth);
}
