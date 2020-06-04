
/*
 * s.c sierpinski carpet code taken from carpetrosettacode.org
 * Modified by Dennis Clarke to remove the horrific CaMeLCAse crap
 *
 * Content is available under GNU Free Documentation License 1.2
 *
 * Permission is granted to copy, distribute and/or modify this document
 * under the terms of the GNU Free Documentation License, Version 1.2
 * or any later version published by the Free Software Foundation;
 * with no Invariant Sections, no Front-Cover Texts, and no Back-Cover
 * Texts. A copy of the license is included in the section entitled "GNU
 * Free Documentation License".
 *
 * GNU Free Documentation License
 *
 * Version 1.2, November 2002
 *
 * Copyright (C) 2000,2001,2002  Free Software Foundation, Inc.
 * 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 * Everyone is permitted to copy and distribute verbatim copies
 * of this license document, but changing it is not allowed.
 *
 */

/*
 * The Open Group Base Specifications Issue 6
 * IEEE Std 1003.1, 2004 Edition
 *
 *    An XSI-conforming application should ensure that the feature
 *    test macro _XOPEN_SOURCE is defined with the value 600 before
 *    inclusion of any header. This is needed to enable the
 *    functionality described in The _POSIX_C_SOURCE Feature Test
 *    Macro and in addition to enable the XSI extension.
 */
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
typedef struct s_carpet {
    int dim;      /* dimension             */
    char *data;   /* character data        */
    char **rows;  /* pointers to data rows */
} *carp;
 
/* Clones a tile into larger carpet, or blank if center */
void tile_carpet( carp d, int r, int c, carp tile )
{
    int y0 = tile->dim*r;
    int x0 = tile->dim*c;
    int k,m;
 
    if ((r==1) && (c==1)) {
        for(k=0; k < tile->dim; k++) {
           for (m=0; m < tile->dim; m++) {
               d->rows[y0+k][x0+m] = ' ';
           }
        }
    }
    else {
        for(k=0; k < tile->dim; k++) {
           for (m=0; m < tile->dim; m++) {
               d->rows[y0+k][x0+m] = tile->rows[k][m];
           }
        }
    }
}
 
/* define a 1x1 starting carpet */
static char s1[]= "#";
static char *r1[] = {s1};
static struct s_carpet single = { 1, s1, r1};
 
carp sierpinski( int n )
{
   carp carpet;
   carp subcarpet;
   int row,col, rb;
   int spc_rqrd;
 
   subcarpet = (n > 1) ? sierpinski(n-1) : &single;
 
   carpet = malloc(sizeof(struct s_carpet));
   carpet->dim = 3 * subcarpet->dim;
   spc_rqrd = (2 * subcarpet->dim) * (carpet->dim);
   carpet->data = malloc(spc_rqrd * sizeof(char));
   carpet->rows = malloc(carpet->dim * sizeof(char *));

   for (row=0; row<subcarpet->dim; row++) {
       carpet->rows[row] = carpet->data + row * carpet->dim;
       rb = row+subcarpet->dim;
       carpet->rows[rb] = carpet->data + rb * carpet->dim;
       rb = row + 2 * subcarpet->dim;
       carpet->rows[rb] = carpet->data + row * carpet->dim;
   }
 
   for (col=0; col < 3; col++) {
       /* 2 rows of tiles to copy - third group points to same data a first */
       for (row=0; row < 2; row++) {
           tile_carpet( carpet, row, col, subcarpet );
       }
   }

   if (subcarpet != &single ) {
       free(subcarpet->rows);
       free(subcarpet->data);
       free(subcarpet);
   }
 
   return carpet;

}
 
void carpet_print( FILE *fout, carp this_carp)
{
    char obuf[730];
    int row;
    for (row=0; row < this_carp->dim; row++) {
        strncpy(obuf, this_carp->rows[row], this_carp->dim);
        fprintf(fout, "%s\n", obuf);
    }
    fprintf(fout,"\n");
}
 
int main(int argc, char *argv[])
{
    /*    FILE *f = fopen("sierp.txt","w");   */
    carpet_print(stdout, sierpinski(3));
    /*    fclose(f); */
    return EXIT_SUCCESS;

}

