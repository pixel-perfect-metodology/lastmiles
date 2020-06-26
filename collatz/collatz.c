
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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <sys/utsname.h>

int sysinfo(void);
uint64_t system_memory(void);

int main(int argc, char *argv[]) 
{
    uint64_t path_length, path_height, height_location, number;
    uint64_t c0; /* this is the starting point number entered */

    setlocale( LC_ALL, "C" );
    sysinfo();

    if ( argc > 1 ){
        number = strtoll(argv[1], (char **)NULL, 10);
        if ( number > 6148914691236517204 ) {
            fprintf(stderr,"FAIL : computation outside uint64_t domain.\n");
            return ( EXIT_FAILURE );
        }
        c0 = number;
    } else {
        fprintf(stderr,"FAIL : please enter a starting number.\n");
        return ( EXIT_FAILURE );
    }

    height_location = 0;
    path_height = 0;
    path_length = 0;

    while (number > 1) {
        /* printf ("%12i    ", k); */
        if (number % 2 == 0) {
            number = number / 2;
            /* printf("%12i", number); */
        } else {
            if ( number > 6148914691236517204 ) {
                /*
                 *  echo ' 6148914691236517203 16o p  3 * 1 + pq' | dc
                 *  5555555555555553
                 *  FFFFFFFFFFFFFFFA   <--- this fits uint64_t
                 *
                 */
                fprintf(stderr,"FAIL : computation outside uint64_t domain.\n");
                fprintf(stderr,"     : %12llu at path position %12llu\n",
                                                          number, path_length );
                return ( EXIT_FAILURE );
            }
            number = 3 * number + 1;
            /* printf("%12i", number); */
        }
        /* printf("\n"); */

        path_length = path_length + 1;

        if ( number > path_height ) {
            path_height = number;
            height_location = path_length;
        }

    }

    printf ("%12llu    %12llu", c0, path_length);

    /* we may have the weird situation where the starting number
     * c0 is the largest number ever */
    if ( c0 > path_height ) {
        printf("    %12llu    %12llu\n", 0, c0);
    } else {
        printf("    %12llu    %12llu\n", height_location, path_height);
    }

    return( EXIT_SUCCESS );

}

