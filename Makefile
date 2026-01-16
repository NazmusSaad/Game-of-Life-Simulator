LOADLIBES = -lm -pthread -lpopt
CC = gcc
CFLAGS = -Wall -O3 -march=native

GOL = gol
GOL_OBJS = gol.o life.o lifeseq.o load.o save.o 
INITBOARD = initboard
INITBOARD_OBJS = initboard.o random_bit.o

all: $(GOL) $(INITBOARD)

$(GOL): $(GOL_OBJS) 

$(INITBOARD): $(INITBOARD_OBJS)

lifeseq.o: lifeseq.c life.h util.h

life.o: life.c life.h util.h

load.o: load.c load.h

save.o: save.c save.h

gol.o: gol.c life.h load.h save.h 

initboard.o: initboard.c random_bit.h

random_bit.o: random_bit.c random_bit.h

clean:
	rm -f $(GOL) $(GOL_OBJS) $(INITBOARD) $(INITBOARD_OBJS)
