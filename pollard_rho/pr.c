
/*
 * pr.c Pollard Rho Algorithm
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

/*********************************************************************
 * The Open Group Base Specifications Issue 6
 * IEEE Std 1003.1, 2004 Edition
 *
 *    An XSI-conforming application should ensure that the feature
 *    test macro _XOPEN_SOURCE is defined with the value 600 before
 *    inclusion of any header. This is needed to enable the
 *    functionality described in The _POSIX_C_SOURCE Feature Test
 *    Macro and in addition to enable the XSI extension.
 *
 *********************************************************************/
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

/* For the Pollard Rho factorization algorithm please
 * see page 976 of the "CLRS" Algorithms textbook.
 *
 * Feel free to test with a prime pair that Jenny gave us :
 *
 *     75261003596099 = 8675309 * 8675311
 *
 * Should factor neatly in about 16 secs or so :
 *
 *     count = 382124  x = 24317867926243  factor = 8675309
 *     A factor of 75261003596099 is 8675309
 *     
 */

uint64_t gcd(uint64_t a, uint64_t b) 
{

    uint64_t remainder;
    uint64_t save_a, save_b;
    uint64_t loop = 0;
    save_a = a;
    save_b = b;

    printf("----   loop %"PRIu64"   ----\n", loop);
    printf("\n------\nEntered into gcd( a = %"PRIu64", b = %"PRIu64")\n", a, b);

    while (b != 0) {

        printf("gcd loop %"PRIu64" :  a = %"PRIu64"  b = %"PRIu64"  and a %% b",
                    loop++, a , b );

        remainder = a % b;
        printf(" remainder = %"PRIu64"\n", remainder);
        a = b;
        b = remainder;
    }
    printf ("gcd( %"PRIu64", %"PRIu64" ) returns %"PRIu64"\n\n------\n", save_a, save_b, a);

    return a;

}

int main (int argc, char *argv[]) 
{

    uint64_t count, number = 10403, loop = 1;
    uint64_t x_fixed = 2, size = 2, x = 2, factor = 1;

    if ( argc>1 ) {

        /* TODO : check that the input is a valid number 
         * someday. */
        number = (uint64_t)strtoll(argv[1], (char **)NULL, 10);
        if ( number > 4294967295 ) {
            fprintf (stderr,"FAIL : number is too large for this trivial\n");
            fprintf (stderr,"     : Pollard Rho implementation.\n");
            return EXIT_FAILURE;
        }

        while (factor == 1) {
            printf("----   loop %"PRIu64"   ----\n", loop);
            printf("\nentering for loop with size = %"PRIu64"\n", size);
            for (count = 1; (count <= size) && (factor <= 1); count++) {
                printf("x = %"PRIu64"\n", x);
                x = (x * x + 1) % number;
                printf("x = (x * x + 1) %% number is now x = %"PRIu64"\n", x);
                printf("x_fixed = %"PRIu64"\n", x_fixed);
                printf("abs(x - x_fixed) = %"PRIu64"\n", abs(x - x_fixed));
                factor = gcd(abs(x - x_fixed), number);
                printf("count = %"PRIu64"  x = %"PRIu64"  factor = %"PRIu64"\n", count, x, factor);
            }
            size = 2 * size;
            x_fixed = x;
            loop = loop + 1;
        }
        printf("A factor of %"PRIu64" is %"PRIu64"\n", number, factor);
        return EXIT_SUCCESS;

    }

    printf("USAGE : %s some_positive_integer\n", argv[0]);

    return EXIT_FAILURE;

}

