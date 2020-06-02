#!/sbin/sh
CC=cc;export CC
LD_OPTIONS=\-R/opt/csw/lib\ \-L/opt/csw/lib
export LD_OPTIONS
CPPFLAGS=\-I/opt/csw/include
export CPPFLAGS

PATH=/opt/studio/SOS8/SUNWspro/bin:/usr/xpg4/bin:/sbin:/bin:/usr/sbin:/usr/bin:/usr/dt/bin:/usr/openwin/bin:/usr/ccs/bin;export PATH
LINES=24;COLUMNS=80;TERM=xterm;LC_ALL=C;EDITOR=vi;LANG=C;export EDITOR LC_ALL TERM COLUMNS LINES LANG

CFLAGS=\-xstrconst\ \-xildoff\ \-xarch\=v9a\ \-Kpic\ \-xlibmil\ \-Xa
export CFLAGS
rm hello.o hello
cc -c -Xc -o hello.o hello.c
cc -Xc -o hello hello.o
file hello
./hello
