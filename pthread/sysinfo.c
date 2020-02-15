
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

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <locale.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <math.h>
#include <fenv.h>
#include <unistd.h>

uint64_t system_memory(void);

int sysinfo(void) {

    struct utsname uname_data;
    uint64_t sysmem = system_memory();
    uint64_t pagesize = (uint64_t)sysconf(_SC_PAGESIZE);
    int fp_round_mode;

    setlocale( LC_MESSAGES, "C" );
    if ( uname( &uname_data ) < 0 ) {
        fprintf ( stderr,
                 "WARNING : Could not attain system uname data.\n" );
        perror ( "uname" );
    } else {
        printf ( "-------------------------------" );
        printf ( "------------------------------\n" );
        printf ( "        system name = %s\n", uname_data.sysname );
        printf ( "          node name = %s\n", uname_data.nodename );
        printf ( "            release = %s\n", uname_data.release );
        printf ( "            version = %s\n", uname_data.version );
        printf ( "            machine = %s\n", uname_data.machine );
        printf ( "          page size = %" PRIu64 "\n", pagesize );
        printf ( "       avail memory = %" PRIu64 "\n", sysmem );
        printf ( "                    = %" PRIu64 " kB\n", sysmem/1024 );
        printf ( "                    = %" PRIu64 " MB\n", sysmem/1048576 );
        /*
         *  this doesn't really work for memory size near GB boundaries
         *
         *  if ( sysmem > ( 1024 * 1048576 ) ) {
         *      printf ( "                    = %" PRIu64 " GB\n",
         *              sysmem/( 1024 * 1048576 ) );
         *  }
         */
        /* get the current floating point rounding mode */
        fp_round_mode = fegetround();
        printf("     fp rounding mode is ");
        switch(fp_round_mode){
            case FE_TONEAREST:
                printf("FE_TONEAREST\n");
                break;
            case FE_TOWARDZERO:
                printf("FE_TOWARDZERO\n");
                break;
            case FE_UPWARD:
                printf("FE_UPWARD\n");
                break;
            case FE_DOWNWARD:
                printf("FE_DOWNWARD\n");
                break;
            default:
                printf("bloody unknown!\n");
                break;
        }

        printf ( "-------------------------------" );
        printf ( "------------------------------" );
    }
    printf ("\n");

    return ( EXIT_SUCCESS );

}

