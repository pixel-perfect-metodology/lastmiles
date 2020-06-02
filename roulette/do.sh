#!/bin/ksh
AS=/usr/ccs/bin/as
export AS

CC=/opt/solarisstudio12.3/bin/cc
export CC

CFLAGS=\-errfmt=error\ \-erroff=%none\ \-errshort=full\ \-errwarn=%all\ \-xstrconst\ \-xildoff\ \-m64\ \-xmemalign=8s\ \-xnolibmil\ \-Xc\ \-xcode=pic32\ \-xregs=no%appl\ \-xlibmieee\ \-mc\ \-g\ \-xs\ \-ftrap=%none\ \-Qy\ \-xbuiltin=%none\ \-xdebugformat=dwarf\ \-xunroll=1\ \-xtarget=ultraT2\ \-xcache=8/16/4:4096/64/16\ \-D_TS_ERRNO\ \-D_POSIX_PTHREAD_SEMANTICS\ \-D_LARGEFILE64_SOURCE
export CFLAGS

unset CONFIG_SHELL

CXX=/opt/solarisstudio12.3/bin/CC
export CXX

CXXFLAGS=\-errfmt=error\ \-erroff=%none\ \-errshort=full\ \-errwarn=%all\ \-xstrconst\ \-xildoff\ \-m64\ \-xmemalign=8s\ \-xcode=pic32\ \-xregs=no%appl\ \-xlibmieee\ \-mc\ \-g\ \-xs\ \-xunroll=1\ \-xtarget=ultraT2\ \-xcache=8/16/4:4096/64/16\ \-D_TS_ERRNO\ \-D_POSIX_PTHREAD_SEMANTICS\ \-D_LARGEFILE64_SOURCE
export CXXFLAGS

EDITOR=/usr/xpg4/bin/vi
export EDITOR

LANG=C
export LANG

LC_ALL=C
export LC_ALL

LC_COLLATE=C
export LC_COLLATE

LC_CTYPE=C
export LC_CTYPE

LC_MESSAGES=C
export LC_MESSAGES

LC_MONETARY=C
export LC_MONETARY

LC_NUMERIC=C
export LC_NUMERIC

LC_TIME=C
export LC_TIME

LD=/usr/ccs/bin/ld
export LD

MANPATH=/usr/share/man:/usr/X11/share/man
export MANPATH

PAGER=/usr/xpg4/bin/more
export PAGER

PATH=/usr/local/bin:/usr/xpg6/bin:/usr/xpg4/bin:/usr/ccs/bin:/opt/solarisstudio12.3/bin:/usr/bin:/sbin:/bin:/usr/sbin:/usr/dt/bin:/usr/openwin/bin:/opt/schily/bin
export PATH

VISUAL=/usr/xpg4/bin/vi
export VISUAL

cc $CFLAGS -o roul roul.c

