
CC = gcc
LD = gcc
CYTHON = cython

CC_FLAGS = -shared -pthread -fPIC -fwrapv -O3 -Wall -fno-strict-aliasing -I/usr/include/python2.5
LD_FLAGS = -o

MODULE_SO = rsvd/rsvd.so

rsvd.so : rsvd.c
	$(CC) $(CC_FLAGS) $(LD_FLAGS) $(MODULE_SO) rsvd/rsvd.c

rsvd.c : rsvd/rsvd.pyx
	$(CYTHON) rsvd/rsvd.pyx
	./instrument.py rsvd/rsvd.c

all : rsvd.so

clean : 
	rm $(MODULE_SO)
        rm *.pyc
        rm rsvd/*.pyc

cleancython : 
	rm rsvd/rsvd.c

tar : clean
	tar -cf $(ARCHIVE) --exclude-vcs -X *.tmp
