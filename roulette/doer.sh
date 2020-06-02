#!/bin/ksh
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

PATH=/usr/xpg6/bin:/usr/xpg4/bin:/usr/ccs/bin:/opt/solarisstudio12.3/bin:/usr/bin:/sbin:/bin:/usr/sbin:/usr/dt/bin:/usr/openwin/bin:/opt/schily/bin
export PATH

SHELL=/bin/ksh
export SHELL

CC=/opt/solarisstudio12.3/bin/cc
export CC

CXX=/opt/solarisstudio12.3/bin/CC
export CXX

EDITOR=/usr/xpg4/bin/vi
export EDITOR

# LD_DEBUG=files
# export LD_DEBUG
unset LD_DEBUG

PAGER=/usr/xpg4/bin/more
export PAGER

unset AR

AS=/usr/ccs/bin/as
export AS

NM=/usr/xpg4/bin/nm\ \-p
export NM

unset OBJDUMP
unset OBJCOPY
unset RANLIB
unset SRC
unset STRIP

VISUAL=/usr/xpg4/bin/vi
export VISUAL

unset LD_RUN_PATH

LD=/usr/ccs/bin/sparcv9/ld
export LD

unset LD_LIBRARY_PATH

unset M4

unset SED

unset GREP

unset PKG_CONFIG_PATH

CFLAGS=\-errfmt=error\ \-erroff=%none\ \-errshort=full\ \-errwarn=%all\ \
\-xstrconst\ \-xildoff\ \-m64\ \-xmemalign=8s\ \-xnolibmil\ \-Xc\ \
\-xcode=pic32\ \-xregs=no%appl\ \-xlibmieee\ \-mc\ \-g\ \-xs\ \
\-ftrap=%none\ \-Qy\ \-xbuiltin=%none\ \-xdebugformat=dwarf\ \-xunroll=1\ \
\-xtarget=ultraT2\ \-xcache=8/16/4:4096/64/16\ \
\-D_TS_ERRNO\ \-D_POSIX_PTHREAD_SEMANTICS\ \-D_LARGEFILE64_SOURCE
export CFLAGS

unset LD_OPTIONS

unset CPPFLAGS

unset LD_RUN_PATH

# /opt/solarisstudio12.3/bin/c99 $CFLAGS -c -o genrand.o genrand.c
/opt/solarisstudio12.3/bin/c99 $CFLAGS -c -o r.o r.c
/opt/solarisstudio12.3/bin/c99 $CFLAGS -o r r.o

ls -l r
rm r.o

/opt/schily/bin/mdigest -a sha256 r


