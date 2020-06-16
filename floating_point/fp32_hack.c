
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>

int main ( int argc, char **argv )
{

    float fp32 = 36.584f;
    double fp64 = 36.584;
    long double fp128 = 36.584L;

    fprintf(stdout,"\n float fp32 = %g\n", fp32 );
    fprintf(stdout," double fp64 = %-+22.16e\n", fp64 );
    fprintf(stdout," long double fp128 = %-+40.34Le\n", fp128 );

    fp32 = 0.1f;
    fp64 = 0.1;
    fp128 = 0.1L;

    fprintf(stdout,"\n float fp32 = %g\n", fp32 );
    fprintf(stdout," double fp64 = %-+22.16e\n", fp64 );
    fprintf(stdout," long double fp128 = %-+40.34Le\n", fp128 );


    fp32 = (float) 1.0000001192092895507812500L;
    fprintf(stdout," 1.000000119209289550781250   = %-+32.26e\n", fp32 );

    fp32 = (float) 1.0000000596046447753906250L;
    fprintf(stdout," 1.0000000596046447753906250  = %-+32.26e\n", fp32 );

    fp32 = (float) 1.000000029802322387695312500L;
    fprintf(stdout," 1.00000002980232238769531250 = %-+32.26e\n", fp32 );

    return EXIT_SUCCESS;

}

