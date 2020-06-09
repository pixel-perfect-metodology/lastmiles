#!/bin/sh
./mpfr_ver 332193 | grep '^atan' | cut -c22-100022 > foo.dat
/usr/bin/printf '0 '; cat foo.dat | grep -o '0' | wc -l
/usr/bin/printf '1 '; cat foo.dat | grep -o '1' | wc -l
/usr/bin/printf '2 '; cat foo.dat | grep -o '2' | wc -l
/usr/bin/printf '3 '; cat foo.dat | grep -o '3' | wc -l
/usr/bin/printf '4 '; cat foo.dat | grep -o '4' | wc -l
/usr/bin/printf '5 '; cat foo.dat | grep -o '5' | wc -l
/usr/bin/printf '6 '; cat foo.dat | grep -o '6' | wc -l
/usr/bin/printf '7 '; cat foo.dat | grep -o '7' | wc -l
/usr/bin/printf '8 '; cat foo.dat | grep -o '8' | wc -l
/usr/bin/printf '9 '; cat foo.dat | grep -o '9' | wc -l
