#!/bin/bash

unset LD_FLAGS

unset AR
unset AS
unset BUILD
unset CC
unset CFLAGS
unset CONFIG_SHELL
unset CPPFLAGS
unset CXX
unset CXXFLAGS
unset EDITOR
unset GREP
unset JRE_HOME
unset LANG
unset LC_ALL
unset LC_COLLATE
unset LC_CTYPE
unset LC_MESSAGES
unset LC_MONETARY
unset LC_NUMERIC
unset LC_TIME
unset LD
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
unset PKG_CONFIG_PATH
unset SED
unset SHELL
unset SRC
unset STD_CDEFINES
unset VISUAL
unset XTERM_LOCALE

unset LANG

unset LC_ALL

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

CFLAGS=\-std=iso9899:1999\ \-m64\ \-g\ \-march=opteron\ \
\-Wl,-rpath=/opt/bw/lib,--enable-new-dtags\ \-fno-builtin\ \
\-malign-double\ \-mpc80
export CFLAGS

unset CXXFLAGS

LDFLAGS=\-Wl,-rpath=/opt/bw/lib,--enable-new-dtags
export LDFLAGS

LD_OPTIONS=\-L/opt/bw/lib
export LD_OPTIONS

LD_RUN_PATH=/opt/bw/lib
export LD_RUN_PATH

unset M4
unset MAKE

MANPATH=/usr/local/share/man:/usr/local/man:/usr/local/share/man:/usr/local/man:/usr/share/man:/opt/schily/share/man
export MANPATH

if [ -h /bin ]; then
    PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/opt/schily/bin; export PATH
else
    PATH=/usr/local/bin:/usr/local/sbin:/sbin:/bin:/usr/sbin:/usr/bin:/opt/schily/bin; export PATH
fi

PKG_CONFIG_PATH=/opt/bw/lib/pkgconfig
export PKG_CONFIG_PATH

RUNPATH=/opt/bw/lib
export RUNPATH

SHELL=/bin/bash
export SHELL

unset SRC

TZ=GMT0
export TZ

XTERM_LOCALE=C; export XTERM_LOCALE

unset LD_LIBRARY_PATH
unset CFLAGS_SOCKETS
unset POSIXLY_CORRECT
unset PHP

# nix_$ ls -lad /usr/bin/ar /usr/bin/as /bin/grep /usr/bin/ld /usr/bin/nm /usr/bin/make /usr/bin/perl /bin/sed
# -rwxr-xr-x 1 root root  198976 Jan  7  2019 /bin/grep
# -rwxr-xr-x 1 root root  122224 Dec 22  2018 /bin/sed
# lrwxrwxrwx 1 root root      19 Mar 21 14:49 /usr/bin/ar -> x86_64-linux-gnu-ar
# lrwxrwxrwx 1 root root      19 Mar 21 14:49 /usr/bin/as -> x86_64-linux-gnu-as
# lrwxrwxrwx 1 root root      19 Mar 21 14:49 /usr/bin/ld -> x86_64-linux-gnu-ld
# -rwxr-xr-x 1 root root  232032 Jul 28  2018 /usr/bin/make
# lrwxrwxrwx 1 root root      19 Mar 21 14:49 /usr/bin/nm -> x86_64-linux-gnu-nm
# -rwxr-xr-x 2 root root 3197768 Mar 31 11:51 /usr/bin/perl
# 
AR=/usr/bin/ar ; export AR
AS=/usr/bin/as ; export AS
if [ -h /bin ]; then
    GREP=/usr/bin/grep; export GREP
else
    GREP=/bin/grep ; export GREP
fi

LD=/usr/bin/ld ; export LD
NM=/usr/bin/nm\ \-p; export NM
MAKE=/usr/bin/make ; export MAKE
PERL=/usr/bin/perl ; export PERL

if [ -h /bin ]; then
    SED=/usr/bin/sed; export SED
else
    SED=/bin/sed ; export SED
fi

CPPFLAGS=\-I/opt/bw/include\ \-D_POSIX_PTHREAD_SEMANTICS\ \-D_TS_ERRNO
export CPPFLAGS

unset CXXFLAGS
unset LD_LIBRARY_PATH
unset LIBTOOL
unset CFLAGS_SOCKETS
unset POSIXLY_CORRECT
unset PHP


if [ -x /usr/bin/gcc ]; then
    CC=/usr/bin/gcc ; export CC
fi

if [ -x /usr/bin/gcc-8 ]; then
    CC=/usr/bin/gcc-8 ; export CC
fi

if [ -x /usr/bin/gcc-9 ]; then
    CC=/usr/bin/gcc-8 ; export CC
fi

/usr/bin/printf "INFO : CC set to "
echo $CC

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

$CC $CFLAGS $CPPFLAGS -I/opt/bw/include -c -o pr.o pr.c
$CC $CFLAGS $CPPFLAGS -I/opt/bw/include -c -o pr_quiet.o pr_quiet.c
$CC $CFLAGS $CPPFLAGS -I/opt/bw/include -c -o pr_mpfr.o pr_mpfr.c
$CC $CFLAGS $CPPFLAGS -I/opt/bw/include -c -o pr_mpfr_quiet.o pr_mpfr_quiet.c
$CC $CFLAGS $CPPFLAGS -I/opt/bw/include -c -o mpfr_set_str.o mpfr_set_str.c 

ls -lapb --full-time *.o
/usr/bin/printf "\n"

$CC $CFLAGS $CPPFLAGS -L/opt/bw/lib -o pr_mpfr pr_mpfr.o -lgmp -lm -lmpfr
$CC $CFLAGS $CPPFLAGS -L/opt/bw/lib -o pr_mpfr_quiet pr_mpfr_quiet.o -lgmp -lm -lmpfr
$CC $CFLAGS $CPPFLAGS -L/opt/bw/lib -o mpfr_set_str mpfr_set_str.o -lgmp -lm -lmpfr

$CC $CFLAGS $CPPFLAGS -o pr pr.o
$CC $CFLAGS $CPPFLAGS -o pr_quiet pr_quiet.o -lm

ls -lapb --full-time pr_mpfr pr_mpfr_quiet mpfr_set_str pr pr_quiet
