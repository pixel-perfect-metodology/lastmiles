
#define _XOPEN_SOURCE 600

#include <stdio.h>

int main(int argc, char **argv)
{

    /* we need an integer value for fibonacci element 0 */
    int f0   = 0;

    /* use the printf function as defined in header file stdio.h */



         printf      (  "hello twitch world   "      )

                        ;






    /* use a format of %i for simple signed integer and 
     * add a newline control character after it. Ensure 
     * that the second argument to this function call is
     * the thing we want printed as an integer */
    printf (   "%i\n"     , f0      )     ;





    /* this is nearly the same thing again but with extra 
     * decoration */
    printf("   0 : f(   0) = %i\n", f0 );

    int f1 = 1;

    int f2;


barf:

    /* here comes an addition statement which is really
     * an assignment of the right expression value to 
     * the left thing on the left of the equals */

          f2    =    f0    +    f1

                                 ;




    /* printout the result f2  */
    printf( "%i\n", f2 );

    /* we want to add  f1 and f2 together but we 
       also will stick with just the variables 
        f0 and f1 and f2 as that is all we have. 
          */


    goto barf ;

    
         return ( 42 )    ; 



    
}


