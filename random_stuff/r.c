
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define RLIM 1024
#define FNAME "/dev/random"

int     main(int argc, char *argv[])
{
    FILE  * fp;
    char    fname[PATH_MAX];
    char    error_buff[255];
    int     i, j, fflush_err, oddball = 0;
    float   rval;

    if ( argc > 1 ) {
        if ( strlen(argv[1]) > PATH_MAX ) {
            fprintf( stderr, "%s: nice try kiddo.\n", argv[0] );
            fprintf( stderr, "%s: max filename length is %i\n", PATH_MAX );
            fprintf( stderr, "%s: ABORTING\n", argv[0] );
            return EXIT_FAILURE;
        }
        strcpy ( fname, argv[1] );
    } else {
        strcpy ( fname, "/dev/random" );
    }

    if ( ( fp = fopen( fname, "r" ) ) == NULL ) {
        fprintf( stderr, "%s: can't read read from %s\n", argv[0], fname);
        fprintf( stderr, "%s: ABORTING\n", argv[0] );
        perror( error_buff );
        exit( EXIT_FAILURE );
    } else {
        for ( i = 0; i < RLIM; i++) {
            j = getc(fp);
            if (argc > 2) {
                printf ( "Byte %4i is %3i (", i, j );
                if (j % 2) {
                    printf (" odd)");
                } else {
                    printf ("even)");
                }
                rval = ( (float) j / (float) 256);
                printf ( " and float is %.8f\n", rval );
            }
            oddball += j % 2 ? 0 : 1;
        }

        fflush_err = fflush(fp);
        if (fflush_err != 0) {
            fprintf(stderr, "fflush error %i", fflush_err);
            exit( EXIT_FAILURE );
        }
        fclose(fp);
        printf ( "%i bytes read and %i were odd.\n", RLIM, oddball );
    }
    return EXIT_SUCCESS;
}

