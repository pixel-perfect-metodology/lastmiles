#!/bin/sh
# LD_LIBRARY_PATH=/opt/bw/lib:/usr/local/lib ./mpfr_ver 3322  | grep '^atan' | cut -c22-1022 > pi1000.dat
/usr/bin/printf '0 '; cat pi_10mil.dat | /usr/local/bin/grep -o '0' | wc -l
/usr/bin/printf '1 '; cat pi_10mil.dat | /usr/local/bin/grep -o '1' | wc -l
/usr/bin/printf '2 '; cat pi_10mil.dat | /usr/local/bin/grep -o '2' | wc -l
/usr/bin/printf '3 '; cat pi_10mil.dat | /usr/local/bin/grep -o '3' | wc -l
/usr/bin/printf '4 '; cat pi_10mil.dat | /usr/local/bin/grep -o '4' | wc -l
/usr/bin/printf '5 '; cat pi_10mil.dat | /usr/local/bin/grep -o '5' | wc -l
/usr/bin/printf '6 '; cat pi_10mil.dat | /usr/local/bin/grep -o '6' | wc -l
/usr/bin/printf '7 '; cat pi_10mil.dat | /usr/local/bin/grep -o '7' | wc -l
/usr/bin/printf '8 '; cat pi_10mil.dat | /usr/local/bin/grep -o '8' | wc -l
/usr/bin/printf '9 '; cat pi_10mil.dat | /usr/local/bin/grep -o '9' | wc -l
