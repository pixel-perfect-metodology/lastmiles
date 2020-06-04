

/*
 * sieve.c Demo Sieve of Eratosthenes
 * Copyright (C) Dennis Clarke 2019
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * https://www.gnu.org/licenses/gpl-3.0.txt
 */


/*
 * The Open Group Base Specifications Issue 6
 * IEEE Std 1003.1, 2004 Edition
 *
 *    An XSI-conforming application should ensure that the feature
 *    test macro _XOPEN_SOURCE is defined with the value 600 before
 *    inclusion of any header. This is needed to enable the
 *    functionality described in The _POSIX_C_SOURCE Feature Test
 *    Macro and in addition to enable the XSI extension.
 */
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[])
{
    int limit, j, k, pcnt, p[10000];
    double estimate;

    if ( argc < 2 ){
        fprintf(stderr,"FAIL : please provide an upper limit\n");
        return(EXIT_FAILURE);
    }

    limit = atoi(argv[1]);
    if ( limit > 10000 ){
        fprintf(stderr,"WARN : upper limit is 10000\n");
        limit = 10000;
    }

    if ( limit < 0 ){
        fprintf(stderr,"WARN : positive numbers please!\n");
        limit = -1 * limit;
    }

    if ( limit < 100 ){
        fprintf(stderr,"WARN : at least 100 please!\n");
        limit = 100;
    }

    for ( k=2; k<limit; k++ ) {
        p[k]=k;
    }

    for ( k=2; k<limit; k++ )
        if ( p[k] > 0 )
            for ( j=k+k; j<limit; j=j+k ) p[j] = 0;
 
    pcnt = 0;
    for ( k=2; k<limit; k++ )
        if ( p[k] > 0 ) {
            printf("    %02i    %i\n", k, p[k]);
            pcnt += 1;
        }
    printf("\n    Number of primes is %i\n", pcnt);

    estimate = ((double)limit) / log((double)limit);

    printf("\n    Estimate n/log(n) = %g\n", estimate);

    return EXIT_SUCCESS;

}

