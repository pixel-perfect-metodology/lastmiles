
    setlocale( LC_ALL, "C" );

    /* Get the REALTIME_CLOCK time in a timespec struct */
    if ( clock_gettime( CLOCK_REALTIME, &now_time ) == -1 ) {
        /* We could not get the clock. Bail out. */
        fprintf(stderr,"ERROR : could not attain CLOCK_REALTIME\n");
        return(EXIT_FAILURE);
    } else {
        /* call srand48() with the sub-second time data */
        srand48( (long) now_time.tv_nsec );
    }
    sysinfo();

    errno = 0;
    if ( ( argc < 6 ) && ( argc > 1 ) ) {
        fprintf(stderr,"FAIL : insufficient arguments provided\n");
        fprintf(stderr,"     : usage %s bail_out_integer \\\n",argv[0]);
        fprintf(stderr,"     :          magnify_integer \\\n");
        fprintf(stderr,"     :          double_real \\\n");
        fprintf(stderr,"     :          double_imaginary\\\n");
        fprintf(stderr,"     :          pthread_count\n");
        fprintf(stderr,"     : quitting.\n");
        return ( EXIT_FAILURE );
    } else if ( argc >= 6 ) {
        /* TODO
         * check if the first char in argv[1] is a letter 'p' and then
         * assume the remaining digits represent a power of 2 */

        candidate_int = (int)strtol(argv[1], (char **)NULL, 10);
        if ( ( errno == ERANGE ) || ( errno == EINVAL ) ){
            fprintf(stderr,"FAIL : bail_out_integer not understood\n");
            perror("     ");
            return ( EXIT_FAILURE );
        }
        if ( ( candidate_int < 256 ) || ( candidate_int > 1048576 ) ){
            fprintf(stderr,"WARN : mandlebrot bail out is unreasonable\n");
            fprintf(stderr,"     : we shall assume 4096 and proceed.\n");
            mand_bail = (uint32_t)4096;
        } else {
            mand_bail = (uint32_t)candidate_int;
        }

        /* TODO fix this to use SCNu64 macro with uint64_t type */
        if ( sscanf( argv[2], "%lld", &candidate_magnify ) == 0 ) {
            fprintf(stderr,"INFO : magnify not understood as long long int\n");

            candidate_int = (int)strtol(argv[2], (char **)NULL, 10);
            if ( ( errno == ERANGE ) || ( errno == EINVAL ) ){
                fprintf(stderr,"FAIL : magnify_integer not understood\n");
                perror("     ");
                return ( EXIT_FAILURE );
            }
            if ( ( candidate_int < 1 ) || ( candidate_int > ( 1<<30 ) ) ){
                fprintf(stderr,"WARN : magnify_integer is unreasonable\n");
                fprintf(stderr,"     : we shall assume 1 and proceed.\n");
                magnify = 1.0;
            } else {
                magnify = (double)candidate_int;
            }
        } else {
            fprintf(stderr,"INFO : magnify accepted as long long int\n");
            magnify = (double)candidate_magnify;
        }

        errno = 0;
        feclearexcept(FE_ALL_EXCEPT);
        candidate_double = strtod(argv[3], (char **)NULL);
        fpe_raised = fetestexcept(FE_ALL_EXCEPT);
        if (fpe_raised!=0){
            printf("INFO : FP Exception raised is");
            if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
            if ( fpe_raised & FE_DIVBYZERO ) printf(" FE_DIVBYZERO");
            if ( fpe_raised & FE_UNDERFLOW ) printf(" FE_UNDERFLOW");
            if ( fpe_raised & FE_OVERFLOW ) printf(" FE_OVERFLOW");
            if ( fpe_raised & FE_INVALID ) printf(" FE_INVALID");
            if ( fpe_raised & FE_INEXACT ) printf(" FE_INEXACT");
            printf("\n");
        }
        if ( fpe_raised & FE_INEXACT ) {
            printf("real : Perfectly safe to ignore FE_INEXACT\n");
        }
        if ( ( errno == ERANGE ) || ( errno == EINVAL ) ){
            fprintf(stderr,"FAIL : double real coordinate not understood\n");
            perror("     ");
            return ( EXIT_FAILURE );
        }
        if ( !isnormal(candidate_double) && ( candidate_double != 0.0 ) ) {
            fprintf(stderr,"FAIL : double real coordinate is not normal\n");
            fprintf(stderr,"     : looks like %-+18.12e\n", candidate_double);
            return ( EXIT_FAILURE );
        }
        feclearexcept(FE_ALL_EXCEPT);
        if ( ( candidate_double < -2.0 ) || ( candidate_double > 2.0 ) ){
            fprintf(stderr,"WARN : double real coordinate is out of range\n");
            fprintf(stderr,"     : value seen = %-+18.12e\n", candidate_double );
            fprintf(stderr,"     : we shall assume zero.\n");
            real_translate = 0.0;
        } else {
            real_translate = candidate_double;
        }

        errno = 0;
        feclearexcept(FE_ALL_EXCEPT);
        candidate_double = strtod(argv[4], (char **)NULL);
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
            printf("imag : Perfectly safe to ignore FE_INEXACT\n");
        }
        if ( ( errno == ERANGE ) || ( errno == EINVAL ) ){
            fprintf(stderr,"FAIL : double imaginary coordinate not understood\n");
            perror("     ");
            return ( EXIT_FAILURE );
        }
        if ( !isnormal(candidate_double) && ( candidate_double != 0.0 ) ) {
            fprintf(stderr,"FAIL : double imaginary coordinate is not normal\n");
            fprintf(stderr,"     : looks like %-+18.12e\n", candidate_double);
            return ( EXIT_FAILURE );
        }
        feclearexcept(FE_ALL_EXCEPT);
        if ( ( candidate_double < -2.0 ) || ( candidate_double > 2.0 ) ){
            fprintf(stderr,"WARN : double imaginary coordinate is out of range\n");
            fprintf(stderr,"     : value seen = %-+18.12e\n", candidate_double );
            fprintf(stderr,"     : we shall assume zero.\n");
            imag_translate = 0.0;
        } else {
            imag_translate = candidate_double;
        }

        candidate_int = (int)strtol(argv[5], (char **)NULL, 10);
        if ( ( errno == ERANGE ) || ( errno == EINVAL ) ){
            fprintf(stderr,"FAIL : pthread_limit not understood\n");
            perror("     ");
            return ( EXIT_FAILURE );
        }
        if ( ( candidate_int < 1 ) || ( candidate_int > 64 ) ){
            fprintf(stderr,"WARN : pthread_limit is unreasonable\n");
            fprintf(stderr,"     : we shall assume 1 and proceed.\n");
            pthread_limit = 1;
        } else {

            if ( candidate_int > 1 ) {

                /* snazzy little bit shifting and counting follows
                 * where this is not at all efficient but is sort of
                 * fun */

                k = 0; /* number of '1' bits in candidate_int */
                j = candidate_int;
                p = 0; /* bit position being tested */
                while (j) {
                    if ( j & 1 ) { /* test the LSB position */
                        k += 1;    /* count the '1' bit */
                    }
                    j = j >> 1;    /* shift left */
                    p += 1;        /* keep track of the bit position */
                }
                if ( k > 1 ) {
                    fprintf(stderr,"WARN : pthread_limit is not a perfect\n");
                    fprintf(stderr,"     : power of two. We shall assume\n");
                    pthread_limit = 1 << ( p - 1 );
                    fprintf(stderr,"     : %i POSIX thread(s).\n", pthread_limit);
                } else {
                    pthread_limit = candidate_int;
                }

            } else {
                pthread_limit = 1;
            }
        }

    } else {
        fprintf(stderr,"WARN : No arguments received thus we have\n");
        fprintf(stderr,"     : some hard coded values ... enjoy.\n");
        mand_bail = 4096;
        magnify = 1.0;
        real_translate = 0.0;
        imag_translate = 0.0;
    }

    printf("\n    mand_bail = %i\n", mand_bail);
    printf("pthread_limit = %i\n", pthread_limit);

    printf("    translate = ( %-+18.14e , %-+18.14e )\n",
                                      real_translate, imag_translate );

    printf("      magnify = %-+18.12e\n\n", magnify );


