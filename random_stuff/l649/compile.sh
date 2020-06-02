#!/bin/ksh
if [ -d /usr/xpg4/bin ]; then
    PATH=/usr/xpg4/bin:/usr/ccs/bin:/usr/bin:/opt/studio/SOS11/SUNWspro/bin:/sbin:/bin:/usr/sbin:/usr/dt/bin:/usr/openwin/bin:/opt/schily/bin
    export PATH
fi

if [ -d /usr/xpg6/bin ]; then
    PATH=/usr/xpg6/bin:/usr/xpg4/bin:/usr/ccs/bin:/opt/studio/SOS11/SUNWspro/bin:/usr/bin:/sbin:/bin:/usr/sbin:/usr/dt/bin:/usr/openwin/bin:/opt/schily/bin
    export PATH
fi

CFLAGS=\-erroff\ \-xstrconst\ \-xildoff\ \-xarch=v9\ \-xmemalign=8s\ \-xnolibmil\ \-Xc\ \-xcode=pic32\ \-xregs=no%appl\ \-xlibmieee\ \-mc\ \-g\ \-xs\ \-D_TS_ERRNO\ \-Dsun=1\ \-Dunix=1\ \-Dsparc=1\ \-D__sun\ \-D__unix\ \-D__SUNPRO_C=0x580\ \-D__SunOS_5_8\ \-D__sparc\ \-D__SVR4\ \-ftrap=%none\ \-Qy\ \-V\ \-xbuiltin=%none\ \-xdebugformat=dwarf\ \-xtime\ \-xunroll=1\ \-DSOLARIS2=8\ \-D_POSIX_PTHREAD_SEMANTICS\ \-D_LARGEFILE64_SOURCE
export CFLAGS

CC=/opt/studio/SOS11/SUNWspro/bin/cc
export CC

CXX=/opt/studio/SOS11/SUNWspro/bin/CC
export CXX

EDITOR=/usr/xpg4/bin/vi
export EDITOR

LANG=C;LC_ALL=C;LC_COLLATE=C;LC_CTYPE=C;LC_MESSAGES=C;LC_MONETARY=C;LC_NUMERIC=C;LC_TIME=C;LD=/usr/ccs/bin/ld
export LANG LC_ALL LC_COLLATE LC_CTYPE LC_MESSAGES LC_MONETARY LC_NUMERIC LC_TIME LD

# LD_DEBUG=files
# export LD_DEBUG

PAGER=/usr/xpg4/bin/more
export PAGER

# AR=/opt/csw/bin/gar
unset AR
# AS=/opt/csw/bin/gas
unset AS

EDITOR=/usr/xpg4/bin/vi
export EDITOR

# NM=/opt/csw/bin/gnm
unset NM

# OBJDUMP=/opt/csw/bin/gobjdump
unset OBJDUMP

# OBJCOPY=/opt/csw/bin/objcopy
unset OBJCOPY

# RANLIB=/opt/csw/bin/granlib
unset RANLIB

SRC=/shared/gar/src
export SRC

# STRIP=/opt/csw/bin/gstrip
unset STRIP

unset LD_DEBUG

VISUAL=/usr/xpg4/bin/vi
export VISUAL

CC=/opt/studio/SOS11/SUNWspro/bin/cc
export CC
VISUAL=/usr/xpg4/bin/vi
export VISUAL
unset LD_LIBRARY_PATH
unset LD_RUN_PATH

unset LD_OPTIONS
unset CPPFLAGS
unset CXXFLAGS

/bin/printf "\nCC    : ---------------------------------------\n"
/opt/studio/SOS11/SUNWspro/bin/cc -H -\# -D_TS_ERRNO -Dsun=1 -Dunix=1 -Dsparc=1 -D__sun -D__unix -D__SUNPRO_C=0x580 -D__SunOS_5_8 -D__sparc -D__SVR4 -DSOLARIS2=8 -D_POSIX_PTHREAD_SEMANTICS -D_LARGEFILE64_SOURCE -erroff=%all -g -H -mc -V -Xc -xarch=v9 -xcode=pic32 -xdebugformat=dwarf -xlibmieee -xmemalign=8s -xregs=no%appl -xs -xstrconst -xtime -xunroll=1 -c -o lot649.o lot649.c
/bin/printf "\nLINK  : ---------------------------------------\n"
/opt/studio/SOS11/SUNWspro/bin/cc -D_TS_ERRNO -Dsun=1 -Dunix=1 -Dsparc=1 -D__sun -D__unix -D__SUNPRO_C=0x580 -D__SunOS_5_8 -D__sparc -D__SVR4 -DSOLARIS2=8 -D_POSIX_PTHREAD_SEMANTICS -D_LARGEFILE64_SOURCE -erroff=%all -g -H -mc -V -Xc -xarch=v9 -xcode=pic32 -xdebugformat=dwarf -xlibmieee -xmemalign=8s -xregs=no%appl -xs -xstrconst -xtime -xunroll=1 -o lot649 lot649.o
/bin/printf "\nDONE  : ---------------------------------------\n"

/bin/printf "\n If that was good try this \n ./lot649 6 8 3 7 16 17 19 21 27 42\n\n"

