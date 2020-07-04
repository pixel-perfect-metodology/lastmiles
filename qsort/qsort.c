
/*
 * qsort.c trivial demonstration of the qsort function
 * Copyright (C) Dennis Clarke 2019
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * https://www.gnu.org/licenses/gpl-3.0.txt
 */

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

/* A trivial function to compare data which happens
 * to just be integers at the moment.  However we
 * could write this to handle nearly anything and
 * that includes the colour of cats.  */
static int
int_compare(const void *p1, const void *p2)
{
    int i = *((int *)p1);
    int j = *((int *)p2);

    if (i > j) return (1);
    if (i < j) return (-1);

    return (0);
}

/* Here we create a trivial array of integers and then
 * call qsort to do what it does. The qsort call uses
 * the function int_compare to determine sort order. */
int main(int argc, char **argv)
{
    int a[14] = { 9, 8, 7, 6, 5, 4, 3,
                  2, 1, 0, -1, -4, 12, -11 };
    size_t j, nelems = sizeof (a) / sizeof(int);

    printf("input  : ");
    for (j = 0; j < nelems; j++) {
        printf("%3d ", a[j]);
    }
    printf("\n");

    qsort((void *)a, nelems, sizeof(int), int_compare);

    printf("output : ");
    for (j = 0; j < nelems; j++) {
        printf("%3d ", a[j]);
    }
    printf("\n");

    return EXIT_SUCCESS;

}

