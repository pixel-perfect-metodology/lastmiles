
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
#include <time.h>
#include <unistd.h>
#include <errno.h>

uint64_t fib(uint64_t n) {
   if(n == 0){
      return 0;
   } else if(n == 1) {
      return 1;
   } else {
      return fib(n-1) + fib(n-2);
   }
}

uint64_t timediff( struct timespec st, struct timespec en );

int main ( int argc, char **argv)
{

    struct timespec t0, t1;
    uint64_t fibber;

    int candidate_int;
    int fib_limit;

    errno = 0;
    if ( argc != 2 ) {
        fprintf(stderr,"FAIL : insufficient arguments provided\n");
        fprintf(stderr,"     : usage %s fibonacci_limit\n",argv[0]);
        return ( EXIT_FAILURE );
    } else {
        candidate_int = (int)strtol(argv[1], (char **)NULL, 10);
        if ( ( errno == ERANGE ) || ( errno == EINVAL ) ){
            fprintf(stderr,"FAIL : fibonacci_limit not understood\n");
            perror("     ");
            return ( EXIT_FAILURE );
        }
        if ( ( candidate_int < 1 ) || ( candidate_int > 56 ) ){
            fprintf(stderr,"WARN : fibonacci_limit is unreasonable\n");
            fprintf(stderr,"     : we shall assume 40 and proceed.\n");
            fib_limit = 40;
        } else {
            fib_limit = candidate_int;
            fprintf(stderr,"INFO : fib_limit is %i\n", fib_limit);
        }
    }

    for ( fibber = 0; fibber < ( fib_limit + 1 ); fibber++ ) {
        clock_gettime( CLOCK_MONOTONIC, &t0 );
        fprintf ( stdout, "fib(%-3" PRIu64 ") = %-12" PRIu64 "    dt = ", fibber, fib(fibber) );
        clock_gettime( CLOCK_MONOTONIC, &t1 );
        fprintf ( stdout, "%12" PRIu64 " nsec\n", timediff( t0, t1 ) );
    }

    return EXIT_SUCCESS;

}

