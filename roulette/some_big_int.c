
/*
 * some_big_int.c Demo a single large unsigned int being messed with.
 * Copyright (C) Dennis Clarke 1989
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

/*
 * this define may not be needed at all
 * #define __STDC_FORMAT_MACROS
 */

#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


int main (int argc, char **argv)
{ 

    uint64_t big_positive_number;

    if ( argc < 2 ) {
        printf ( "usage : %s some_integer\n", argv[0] );
        exit( EXIT_FAILURE );
    }

    /* This will stuff 0xffff ffff ffff ffff into the memory
     * address of the 64-bit variable in spite of the fact
     * that it should be unsigned. We can perform a cast of
     * the literal data element from signed data to unsigned.
     */
    big_positive_number = (uint64_t) -1l;
    printf ("big_positive_number = %"PRIu64"\n", big_positive_number);

    /* now we over write those bits with data from the user */
    big_positive_number = strtol( argv[1], (char **)NULL, 10);
    fprintf ( stderr, "%i : %s\n", errno, strerror(errno) );
    printf ("big_positive_number = %"PRIu64"\n", big_positive_number);

    return ( EXIT_SUCCESS );

}

