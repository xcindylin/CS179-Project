CC          = g++
CFLAGS      = -Wall -ansi -pedantic -ggdb
OBJS        = testgame.o board.o deviceboard.o exampleplayer.o player.o gpuplayer.o decisiontree.o paralleldecisiontree.o node.o devicenode.o

all: testgame
        
testgame: $(OBJS)
	nvcc -arch=sm_20 $(OBJS) -o testgame
        
%.o: %.cpp tree_cuda.cu
	nvcc -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
	rm -f *.o testgame		
