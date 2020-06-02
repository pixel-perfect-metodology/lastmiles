#!/bin/sh
unset AR
unset AS
unset BUILD
unset CC
unset CFLAGS
unset CFLAGS_SOCKETS
unset CONFIG_SHELL
unset CPPFLAGS
unset CXX
unset CXXFLAGS
unset GREP
unset JAVA_HOME
unset JRE_HOME
unset LC_ALL
unset LD
unset LD_FLAGS
unset LD_LIBRARY_PATH
unset LD_OPTIONS
unset LD_RUN_PATH
unset LIBTOOL
unset M4
unset MACHTYPE
unset MAKE
unset MANPATH
unset NM
unset OPENSSL_SOURCE
unset OSTYPE
unset PAGER
unset PERL
unset PHP
unset PKG_CONFIG_PATH
unset POSIXLY_CORRECT
unset SED
unset SHELL
unset SRC
unset STD_CDEFINES

PATH=/usr/bin:/sbin:/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:/opt/schily/bin
export PATH

TMPDIR=/var/tmp/`( /usr/bin/id | /usr/bin/cut -f2 -d\( | /usr/bin/cut -f1 -d\) )`
export TMPDIR

if [ ! -d $TMPDIR ]; then
    /usr/bin/printf "INFO : no TMPDIR exists in /var/tmp/$USERNAME\n"
    mkdir -m 0750 $TMPDIR
    if [ ! -d $TMPDIR ]; then
        /usr/bin/printf "FAIL : could not create a TMPDIR\n"
        exit 1
    fi
    /usr/bin/printf "INFO : new TMPDIR created\n"
else
    chmod 0750 $TMPDIR
    touch $TMPDIR/foo_$$
    if [ ! -f $TMPDIR/foo_$$ ]; then
        /usr/bin/printf "FAIL : could not create a file in TMPDIR\n"
        exit 1
    fi
    rm $TMPDIR/foo_$$
fi

rm *.s *.o *.i *.bc > /dev/null 2>&1

CC=/usr/bin/cc
export CC

LD_FLAGS=\-Wl,-rpath=/home/dclarke/local/lib:/usr/local/lib\ \-L/home/dclarke/local/lib\ \-L/usr/local/lib
export LD_FLAGS

CFLAGS=\-m64\ \-std=c99\ \-fno-builtin\ \-O0\ \-g\ \
\-no-integrated-as\ \-fno-fast-math\ \-Weverything\ \
\-fdiagnostics-format=vi\ \-fno-color-diagnostics
export CFLAGS 

CPPFLAGS=\-D_TS_ERRNO\ \-D_POSIX_PTHREAD_SEMANTICS\ \
\-D_LARGEFILE64_SOURCE\ \-D_XOPEN_SOURCE=600
export CPPFLAGS

rm -f ent.o iso8859.o randtest.o chisq.o ent
$CC $CFLAGS $CPPFLAGS -c ent.c -o ent.o
$CC $CFLAGS $CPPFLAGS -c iso8859.c -o iso8859.o
$CC $CFLAGS $CPPFLAGS -c randtest.c -o randtest.o
$CC $CFLAGS $CPPFLAGS -c chisq.c -o chisq.o
$CC $CFLAGS $CPPFLAGS -o ent ent.o iso8859.o randtest.o chisq.o -lm

RESULT=$?
if [ $RESULT -eq 0 ]; then
    ls -lapb ent
else
    /usr/bin/printf "\nFAIL to compile.\n"
fi
exit $RESULT

