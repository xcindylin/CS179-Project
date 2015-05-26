#ifndef __COMMON_H__
#define __COMMON_H__

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

enum Side { 
    BLACK, WHITE
};

class Move {
   
public:
    int x, y;
    CUDA_CALLABLE_MEMBER Move(int x, int y) {
        this->x = x;
        this->y = y;        
    }
    CUDA_CALLABLE_MEMBER ~Move() {}

    CUDA_CALLABLE_MEMBER int getX() { return x; }
    CUDA_CALLABLE_MEMBER int getY() { return y; }

    CUDA_CALLABLE_MEMBER void setX(int x) { this->x = x; }
    CUDA_CALLABLE_MEMBER void setY(int y) { this->y = y; }
};

#endif
