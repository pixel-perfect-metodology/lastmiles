
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
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <errno.h>

int main(int argc, char **argv)
{

    int limit;
    int fibo_num = 0;
    uint64_t f0   = 0;

    printf("     : f(   0) = %"PRIu64"\n", f0 );

    uint64_t f1 = 1;
    printf("     : f(   1) = %"PRIu64"\n", f1 );

    uint64_t f2;

    if ( argc < 2 ) {
        fprintf ( stderr, "FAIL : gimme a Fibonacci limit\n");
        return ( EXIT_FAILURE );
    }

    int candidate_int = (int)strtol(argv[1], (char **)NULL, 10);

    if ( ( errno == ERANGE ) || ( errno == EINVAL ) ) {
        fprintf(stderr,"FAIL : Fibonacci limit not understood\n");
        perror("     ");
        return ( EXIT_FAILURE );
    }

    if ( ( candidate_int < 3 ) || ( candidate_int > 92 ) ) {
        fprintf(stderr,"WARN : fibonacci limit unreasonable\n");
        return ( EXIT_FAILURE );
    } else {
        limit = candidate_int;
    }

    while ( fibo_num < ( limit - 1 ) ) {
       f2 = f0 + f1;
       fibo_num = fibo_num + 1;
       printf("     : f(%4i) = %"PRIu64"\n", fibo_num +1, f2 );
       f0 = f1;
       f1 = f2;
    }

    return ( 42 )    ; 

}

