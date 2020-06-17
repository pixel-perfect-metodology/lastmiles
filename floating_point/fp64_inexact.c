
/*********************************************************************
 * The Open Group Base Specifications Issue 6
 * IEEE Std 1003.1, 2004 Edition
 *    An XSI-conforming application should ensure that the feature
 *    test macro _XOPEN_SOURCE is defined with the value 600 before
 *    inclusion of any header. This is needed to enable the
 *    functionality described in The _POSIX_C_SOURCE Feature Test
 *    Macro and in addition to enable the XSI extension.
 *********************************************************************/
#define _XOPEN_SOURCE 600

#include <errno.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include <fenv.h>
#pragma STDC FENV_ACCESS ON

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

/* Accept ascii data on the command line and attempt to convert
 * to IEEE 754-2008 floating point FP64 data.
 * 
 * FreeBSD 12 RELEASE triggers a floating point exception 
 * FE_INEXACT on some data that should be accepted perfectly. */

uint64_t system_memory(void);
int sysinfo(void);

int main ( int argc, char *argv[] )
{

    double candidate_double, num;
    int fpe_raised = 0;

    if ( argc < 2 ) {
        fprintf(stderr,"FAIL : provide a decimal number\n");
        return ( EXIT_FAILURE );
    }

    char *buf = malloc((size_t)32);
    if ( buf == NULL ) {
        perror ("malloc!");
        fprintf (stderr,"FAIL : malloc failed for buf\n");
        return(EXIT_FAILURE);
    }

    if ( argc > 3 ) {
        printf ("\nINFO : You suggest a locale of %s\n", argv[2]);
        buf = setlocale ( LC_ALL, argv[2] );
    } else {
        buf = setlocale ( LC_ALL, "POSIX" );
    }

    if ( buf == NULL ) {
        fprintf (stderr,"FAIL : setlocale fail\n");
        return(EXIT_FAILURE);
    }

    sysinfo();

    errno = 0;
    feclearexcept(FE_ALL_EXCEPT);

    candidate_double = strtod(argv[1], (char **)NULL);

    fpe_raised = fetestexcept(FE_ALL_EXCEPT);
    if (fpe_raised!=0){
        printf("INFO : FP Exception raised is");
        if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
        if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
        if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
        if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
        if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
        printf("\n");
    }

    if ( fpe_raised & FE_INEXACT ) {
        printf("WARN : FE_INEXACT returned by strtod()\n");
    }

    if ( ( errno == ERANGE ) || ( errno == EINVAL ) ){
        fprintf(stderr,"FAIL : number not understood\n");
        perror("     ");
        return ( EXIT_FAILURE );
    }

    if ( !isnormal(candidate_double) && ( candidate_double != 0.0 ) ) {
        fprintf(stderr,"FAIL : number is not normal\n");
        fprintf(stderr,"     : looks like %-+22.16e\n", candidate_double);
        return ( EXIT_FAILURE );
    }

    feclearexcept(FE_ALL_EXCEPT);

    num = candidate_double;
    /* slightly wide format spec to see many digits which should
     * be well past the FP64 precision */
    printf ("INFO : seems like a decimal number %-+36.34g\n", num);
    return ( EXIT_SUCCESS );

}

uint64_t system_memory()
{
    /* should return the amount of memory available in bytes */
    long en;
    uint64_t pages, page_size;

    en = sysconf(_SC_PHYS_PAGES);
    if ( en < 0 ){
        perror("sysconf(_SC_PHYS_PAGES) : ");
        exit(EXIT_FAILURE);
    }
    pages = (uint64_t) en;

    page_size = (uint64_t)sysconf(_SC_PAGE_SIZE);
    return ( pages * page_size );
}

int sysinfo(void) {

    struct utsname uname_data;
    uint64_t sysmem = system_memory();
    uint64_t pagesize = (uint64_t)sysconf(_SC_PAGESIZE);
    int fp_round_mode;

    /* guess the architecture endianess? */
    int end_check = 1;
    int little_endian = (*(uint8_t*)&end_check == 1) ? 1 : 0;

    setlocale( LC_MESSAGES, "POSIX" );
    if ( uname( &uname_data ) < 0 ) {
        fprintf ( stderr,
                 "WARNING : Could not attain system uname data.\n" );
        perror ( "uname" );
    } else {
        printf ( "-------------------------------" );
        printf ( "------------------------------\n" );
        printf ( "           system name = %s\n", uname_data.sysname );
        printf ( "             node name = %s\n", uname_data.nodename );
        printf ( "               release = %s\n", uname_data.release );
        printf ( "               version = %s\n", uname_data.version );
        printf ( "               machine = %s\n", uname_data.machine );
        printf ( "             page size = %" PRIu64 "\n", pagesize );
        printf ( "          avail memory = %" PRIu64 "\n", sysmem );
        printf ( "                       = %" PRIu64 " kB\n", sysmem/1024 );
        printf ( "                       = %" PRIu64 " MB\n", sysmem/1048576 );
        /*  does not really work for memory size near GB boundaries
         *  if ( sysmem > ( 1024 * 1048576 ) )
         *      printf ( "                    = %" PRIu64 " GB\n",
         *              sysmem/( 1024 * 1048576 ) );
         */

        printf ( "                endian = ");
        if ( little_endian ) {
            printf ( "little");
        } else {
            printf ( "big");
        }
        printf ( " endian\n" );

        printf ( " sizeof(unsigned long) = %lu\n", sizeof(unsigned long) );
        printf ( "           sizeof(int) = %lu\n", sizeof(int) );
        printf ( "         sizeof(void*) = %lu\n", sizeof(void*) );

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
                printf("unknown!\n");
                break;
        }

        printf ( "-------------------------------" );
        printf ( "------------------------------" );
    }
    printf ("\n");

    return ( EXIT_SUCCESS );

}

