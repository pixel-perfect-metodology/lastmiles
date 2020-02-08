
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main( int gag, char **puke)
{
    fprintf(stderr,"hack : sizeof (unsigned long) = %i\n", sizeof(unsigned long) );
    fprintf(stderr,"hack :      sizeof (uint32_t) = %i\n\n", sizeof(uint32_t) );

    return ( EXIT_SUCCESS );

}


