#!/bin/ksh
cat run_template | sed 's/PROFIT/1000/' > /tmp/run_$$.sh 
chmod 755 /tmp/run_$$.sh 
/tmp/run_$$.sh 
rm /tmp/run_$$.sh 

