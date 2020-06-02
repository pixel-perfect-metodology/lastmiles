#!/bin/ksh 
cc -\# $CFLAGS -c -o spin.o spin.c
cc -\# $CFLAGS -c -o genrand.o genrand.c 
cc -\# $CFLAGS -o spin spin.o genrand.o
