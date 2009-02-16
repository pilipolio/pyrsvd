
CC = gcc
LD = gcc
CYTHON = cython

CC_FLAGS = -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python2.5
LD_FLAGS = -o

MODULE_SO = rsvd.so

rsvd.so : rsvd.c
	$(CC) $(CC_FLAGS) $(LD_FLAGS) $(MODULE_SO) rsvd.c

rsvd.c : rsvd.pyx
	$(CYTHON) rsvd.pyx
	./instrument.py rsvd.c

all : rsvd.so

clean : 
	rm $(MODULE_SO) rsvd.c

cleancython : 
	rm rsvd.c