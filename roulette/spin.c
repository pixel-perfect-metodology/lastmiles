#define _POSIX_SOURCE 1

#include <stdio.h>
#include <stdlib.h>

double genrand();

int main ( int argc, char *argv[])
{

   int j, ball, cell[37];

   for ( j=0; j<50; j++) {
       printf(" %7f", genrand());
       if ( j%8 == 7 ) printf("\n");
   }

   cell[0] = 0;

   ball = 38;

   printf ( "\n\n ball = %2i\n", ball );

   return ( EXIT_SUCCESS );

}
