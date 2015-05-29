#include <cstdio>
#include "tree_cuda.cuh"

__device__ 
void cudaSearch(DeviceNode *node, Side side, Side maximizer, int depth) {
    if (depth == 0) {
       node->setAlpha(node->getScore());
       node->setBeta(node->getScore());
       return;
    }
    DeviceBoard *board = node->getBoard();
    Side oppositeSide = side == BLACK ? WHITE : BLACK;

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
        }
    }

    // thrust::device_vector<Move> moves = board->getMoves(oppositeSide);
    // for (int i = 0; i < moves.size(); i++) {
    //     // create the next child
    //     Move *move = new Move(moves[i].getX(), moves[i].getY());
    //     DeviceBoard *newBoard = board->copy();
    //     newBoard->doMove(move, oppositeSide);
    //     DeviceNode *child = new DeviceNode(move, oppositeSide, maximizer, newBoard);

    //     // pass alpha and beta values down
    //     child->setAlpha(node->getAlpha());
    //     child->setBeta(node->getBeta());

    //     // search child
    //     cudaSearch(child, oppositeSide, maximizer, depth - 1);

    //     if (side == maximizer) {
    //         node->setBeta(min(node->getBeta(), child->getAlpha()));
    //     } else {
    //         node->setAlpha(max(node->getAlpha(), child->getBeta()));
    //     }

    //     delete child;

    //     if (node->getAlpha() >= node->getBeta()) {
    //         return;
    //     }
    // }
}

__global__
void cudaTreeKernel(Move *moves, char *black, char *taken, int *values, Side side, 
    Side maximizer, int alpha, int beta, int depth) {
    // only one thread does high-level tasks
    if (threadIdx.x == 0) {
        // make one new node per block
        Move *move = new Move(moves[blockIdx.x].getX(), moves[blockIdx.x].getY());

        DeviceBoard *newBoard = new DeviceBoard(black, taken);
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
        printf("values: %d index: %d\n", values[blockIdx.x], blockIdx.x);
    }

}

void cudaCallTreeKernel(Move *moves, char *black, char *taken, int *values, Side side, 
    Side maximizer, int alpha, int beta, int numMoves, int depth) {

    printf("hello...\n");

    cudaTreeKernel<<<numMoves, 64>>>(moves, black, taken, values, side, 
       maximizer, alpha, beta, depth);
}
