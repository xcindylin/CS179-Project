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
    search(root, depth);
    // find the actual best move
    vector<Node *> children = root->getChildren();
    Node *best = children[0];
    for (int i = 1; i < children.size(); i++) {
        if (children[i]->getBeta() > best->getBeta()) {
            best = children[i];
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
    Side oppositeSide = startingNode->getSide() == BLACK ? WHITE : BLACK;
    vector<Move> moves = board->getMoves(oppositeSide);
    for (int i = 0; i < moves.size(); i++) {
        // create the next child
        Move move = moves[i];
        Board *newBoard = board->copy();
        newBoard->doMove(&move, oppositeSide);
        Node *child = new Node(&move, oppositeSide, maximizer, newBoard);

        // pass alpha and beta values down
        child->setAlpha(startingNode->getAlpha());
        child->setBeta(startingNode->getBeta());
        startingNode->addChild(child);

        // search child
        search(child, depth - 1);

        if (startingNode->getSide() == maximizer) {
            startingNode->setBeta(min(startingNode->getBeta(), child->getAlpha()));
        } else {
            startingNode->setAlpha(max(startingNode->getAlpha(), child->getBeta()));
        }

        if (startingNode->getAlpha() > startingNode->getBeta()) {
            return;
        }
    }
}