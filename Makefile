CC          = g++
CFLAGS      = -Wall -ansi -pedantic -ggdb

all: testgame
        
testgame: testgame.o board.o exampleplayer.o player.o decisiontree.o node.o
	$(CC) -o $@ $^
        
%.o: %.cpp
	$(CC) -c $(CFLAGS) -x c++ $< -o $@

clean:
	rm -f *.o testgame		
