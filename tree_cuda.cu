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
            if (board->checkMove(move, side)) {
                DeviceBoard *newBoard = board->copy();
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
void cudaTreeKernel(Move *moves, DeviceBoard *board, int *values, Side side, 
    Side maximizer, int alpha, int beta, int depth) {
    // only one thread does high-level tasks
    if (threadIdx.x == 0) {
        // make one new node per block
        Move *move = new Move(moves[blockIdx.x].getX(), moves[blockIdx.x].getY());
        DeviceBoard *newBoard = board->copy();
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
    }

}

void cudaCallTreeKernel(Move *moves, DeviceBoard *board, int *values, Side side, 
    Side maximizer, int alpha, int beta, int numMoves, int depth) {

    cudaTreeKernel<<<numMoves, 64>>>(moves, board, values, side, 
       maximizer, alpha, beta, depth);
}
