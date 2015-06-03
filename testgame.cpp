#include <string>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "common.h"
#include "board.h"
#include "exampleplayer.h"
#include "player.h"
#include "gpuplayer.h"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrchk(cudaEventCreate(&start));       \
      gpuErrchk(cudaEventCreate(&stop));        \
      gpuErrchk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrchk(cudaEventRecord(stop));                     \
      gpuErrchk(cudaEventSynchronize(stop));                \
      gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrchk(cudaEventDestroy(start));                   \
      gpuErrchk(cudaEventDestroy(stop));                    \
  }

// Initialize timers for benchmarking
float cpu_ms = -1;
float gpu_ms = -1;

int main() {
	Board *board = new Board();
    Side turn = BLACK;
    Move *m = NULL;

    Player *player1 = new Player(BLACK);
    ExamplePlayer *player2 = new ExamplePlayer(WHITE);

    cout << "Starting CPU game..." << endl;
    START_TIMER();
    while (!board->isDone()) {
        // get the current player's move
        if (turn == BLACK) {
            m = player1->doMove(m);
        }
        else {  
            m = player2->doMove(m);   
        }

        if (!board->checkMove(m, turn)) {
            cout << "Illegal move made: " << turn << " address: " << m << endl;
        }

        // make move once it is determiend to be legal
        board->doMove(m, turn);

        // switch players
        if (turn == BLACK) {
            turn = WHITE;
        }
        else {
            turn = BLACK;
        }
    }
    STOP_RECORD_TIMER(cpu_ms);

    cout << "CPU Game completed." << endl;
    cout << "Black score: " << board->countBlack() << endl;
    cout << "White score: " << board->countWhite() << endl;

    board = new Board();
    turn = BLACK;
    m = NULL;

    // Run game on GPU here
    GPUPlayer *player3 = new GPUPlayer(BLACK);
    ExamplePlayer *player4 = new ExamplePlayer(WHITE);

    cout << endl << "Starting GPU game..." << endl;
    START_TIMER();
    while (!board->isDone()) {
        // get the current player's move
        if (turn == BLACK) {
            m = player3->doMove(m);
        }
        else { 
            m = player4->doMove(m);   
        }

        if (!board->checkMove(m, turn)) {
            cout << "Illegal move made: " << turn << " address: " << m << endl;
        }

        // make move once it is determiend to be legal
        board->doMove(m, turn);

        // switch players
        if (turn == BLACK) {
            turn = WHITE;
        }
        else {
            turn = BLACK;
        }
    }
    STOP_RECORD_TIMER(gpu_ms);

    cout << "GPU Game completed." << endl;
    cout << "Black score: " << board->countBlack() << endl;
    cout << "White score: " << board->countWhite() << endl;

    cout << endl;
    cout << "CPU time: " << cpu_ms << " milliseconds" << endl;
    cout << "GPU time: " << gpu_ms << " milliseconds" << endl;
    cout << "Speedup factor: " << cpu_ms / gpu_ms << endl << endl;

    return 0;
}
