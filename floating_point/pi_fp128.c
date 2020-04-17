
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

int main(int argc, char *argv[])
{

    long double pi = 3.14159265358979323846264338327950288419716939937510L;

    printf("the sizeof(pi) is %i bytes\n", sizeof(long double) );

    printf("\npi is this  3.14159265358979323846264338");
    printf("327950288419716939937510.....\n\n");

    printf("pi could be %42.38Le\n", pi);

    return ( EXIT_SUCCESS );

}

