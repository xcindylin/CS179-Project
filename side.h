#ifndef __SIDE_H__
#define __SIDE_H__

using namespace std;

class Side {
   
private: 
    Side::Color _color;

public:
    enum Color { 
        WHITE, BLACK
    };

    Side();
    ~Side();

    Side::Color opposite() {
        if (_color == WHITE)
            return BLACK;
        else
            return WHITE;
    }

    string toString() {
        if (_color == WHITE)
            return "White"
        else
            return "Black"
    }

};

#endif
