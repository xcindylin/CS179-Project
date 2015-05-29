#include "paralleldecisiontree.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

ParallelDecisionTree::ParallelDecisionTree(Board *board, Side maximizer) {
    // this is our side
    this->maximizer = maximizer;
    root = new Node(NULL, maximizer == BLACK ? WHITE : BLACK, maximizer, board);
}

ParallelDecisionTree::~ParallelDecisionTree() {
    // free some stuff
}

// DeviceBoard *ParallelDecisionTree::HostToDeviceBoard(Board *board) {
//     DeviceBoard *newDeviceBoard = new DeviceBoard();
//     for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
//         newDeviceBoard->black[i] = board->black[i];
//         newDeviceBoard->taken[i] = board->taken[i];
//     }
//     return newDeviceBoard;
// }

Node *ParallelDecisionTree::getRoot() {
    return root;
}

Move *ParallelDecisionTree::search(Node *startingNode, int depth) {
	if (depth == 0) {
		startingNode->setAlpha(startingNode->getScore());
		startingNode->setBeta(startingNode->getScore());
		return NULL;
	}
	Board *board = startingNode->getBoard();
	Side oppositeSide = startingNode->getSide() == BLACK ? WHITE : BLACK;
	vector<Move> moves = board->getMoves(oppositeSide);

    if (moves.size() == 0) {
        return NULL;
    }
	/* CPU search the first child node */
	Move *move = new Move(moves[0].getX(), moves[0].getY());
    Board *newBoard = board->copy();
    newBoard->doMove(move, oppositeSide);
    Node *child = new Node(move, oppositeSide, maximizer, newBoard);

    // pass alpha and beta values down
    child->setAlpha(startingNode->getAlpha());
    child->setBeta(startingNode->getBeta());

    // search child
    Move *best = search(child, depth - 1);

    // array to store the values of interest of the children
    int *values;
    values = (int *)calloc(moves.size(), sizeof(int));

    if (startingNode->getSide() == maximizer) {
        startingNode->setBeta(min(startingNode->getBeta(), child->getAlpha()));
        values[0] = child->getAlpha();
    } else {
        startingNode->setAlpha(max(startingNode->getAlpha(), child->getBeta()));
        values[0] = child->getBeta();
    }

    delete child;

    /* GPU search the rest of the child nodes */
    int numMoves = moves.size() - 1;
    Move *dev_moves;
    Move *moves_ptr = &moves[1];
    char *dev_black;
    char *dev_taken;
    int *dev_values;

    char *black;
    char *taken;

    black = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));
    taken = (char *) malloc(BOARD_SIZE * BOARD_SIZE * sizeof(char));

    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        black[i] = board->black[i];
        taken[i] = board->taken[i];
    }

    gpuErrchk(cudaMalloc((void **) &dev_moves, numMoves * sizeof(Move)));
    gpuErrchk(cudaMalloc((void **) &dev_black, BOARD_SIZE * BOARD_SIZE * sizeof(char)));
    gpuErrchk(cudaMalloc((void **) &dev_taken, BOARD_SIZE * BOARD_SIZE * sizeof(char)));
    gpuErrchk(cudaMalloc((void **) &dev_values, numMoves * sizeof(int)));

    gpuErrchk(cudaMemcpy(dev_black, board->black, BOARD_SIZE * BOARD_SIZE * sizeof(char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_taken, board->taken, BOARD_SIZE * BOARD_SIZE * sizeof(char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_moves, moves_ptr, numMoves * sizeof(Move), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(dev_values, 0, numMoves * sizeof(int)));

    // call kernel to search the rest of the children in parallel
    cudaCallTreeKernel(dev_moves, dev_black, dev_taken, dev_values, oppositeSide, maximizer, 
    	startingNode->getAlpha(), startingNode->getBeta(), numMoves, depth - 1);

    // copy remaining child values into host array
    gpuErrchk(cudaMemcpy(values + 1, dev_values, numMoves * sizeof(int), cudaMemcpyDeviceToHost));

    cout << "depth: " << depth << endl;
    for (int i = 0; i <= numMoves; i++) {
        cout << values[i] << endl;
    }

    // find the best move
    int index = 0;
    if (startingNode->getSide() == maximizer) {
    	int best = 99999999;
    	for (int i = 0; i <= numMoves; i++) {
    		if (values[i] < best) {
    			best = values[i];
    			index = i;
    		}
    	}
    	startingNode->setBeta(best);
    } else {
    	int best = -99999999;
    	for (int i = 0; i <= numMoves; i++) {
    		if (values[i] > best) {
    			best = values[i];
    			index = i;
    		}
    	}
    	startingNode->setAlpha(best);
    }

    Move *curMove = new Move(moves[index].getX(), moves[index].getY());
    return curMove;
}
