#include "decisiontree.h"

DecisionTree::DecisionTree(Board *board, Side maximizer) {
    // this is our side
    this->maximizer = maximizer;
    root = new Node(NULL, maximizer == BLACK ? WHITE : BLACK, maximizer, board);
}

DecisionTree::~DecisionTree() {
    // free some stuff
}

Move *DecisionTree::findBestMove(int depth) {
    Board *board = root->getBoard();
    vector<Move> moves = board->getMoves(maximizer);
    Node *best = NULL;
    for (int i = 0; i < moves.size(); i++) {
        Move *move = new Move(moves[i].getX(), moves[i].getY());
        Board *newBoard = board->copy();
        newBoard->doMove(move, maximizer);
        Node *child = new Node(move, maximizer, maximizer, newBoard);

        // pass alpha and beta values down
        child->setAlpha(root->getAlpha());
        child->setBeta(root->getBeta());

        // search child
        search(child, depth - 1);

        if (best == NULL || child->getBeta() > best->getBeta()) {
            best = child;
        }
    }
    return best->getMove();
}

void DecisionTree::search(Node *startingNode, int depth) {
    if (depth == 0) {
        startingNode->setAlpha(startingNode->getScore());
        startingNode->setBeta(startingNode->getScore());
        return;
    }
    Board *board = startingNode->getBoard();
    Side oppositeSide = startingNode->getSide() == BLACK ? WHITE : BLACK;
    vector<Move> moves = board->getMoves(oppositeSide);
    for (int i = 0; i < moves.size(); i++) {
        // create the next child
        Move *move = new Move(moves[i].getX(), moves[i].getY());
        Board *newBoard = board->copy();
        newBoard->doMove(move, oppositeSide);
        Node *child = new Node(move, oppositeSide, maximizer, newBoard);

        // pass alpha and beta values down
        child->setAlpha(startingNode->getAlpha());
        child->setBeta(startingNode->getBeta());

        // search child
        search(child, depth - 1);

        if (startingNode->getSide() == maximizer) {
            startingNode->setBeta(min(startingNode->getBeta(), child->getAlpha()));
        } else {
            startingNode->setAlpha(max(startingNode->getAlpha(), child->getBeta()));
        }

        delete child;

        if (startingNode->getAlpha() > startingNode->getBeta()) {
            return;
        }
    }
}
