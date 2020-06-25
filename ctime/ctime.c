
/*
 * ctime.c Extract UNIX ctime etc
 * Copyright (C) Dennis Clarke 1999
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

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <fcntl.h>

int main(int argc, char *argv[])
{
    struct stat status_buffer;
    int status;
    struct timespec start_tv;
    char *c_time_string;

    setlocale ( LC_ALL, "POSIX" );
    if ( clock_gettime( CLOCK_REALTIME, &start_tv ) == -1 ) {
      perror( "clock gettime" );
      return( EXIT_FAILURE );
    }

    c_time_string = ctime( &start_tv.tv_sec );
 
    fprintf ( stderr, "INFO  : current time is %s", c_time_string );

    if (argc>1) {
        status = stat(argv[1], &status_buffer);
    } else {
        fprintf ( stderr, "FAIL : give us a filename\n");
        return( EXIT_FAILURE );
    }

    /* if the stat call worked then we get a buffer with a 
     * pile of good infor in it : 
     *
     *      status_buffer = {
     *          st_dev     = 1640677507086U
     *          st_ino     = 231469791U
     *          st_mode    = 33261U
     *          st_nlink   = 1U
     *          st_uid     = 16411
     *          st_gid     = 20002
     *          st_rdev    = 18446744073709551615U
     *          st_size    = 531
     *
     *          st_atime   = {
     *              __tv_sec  = 1541445959
     *              __tv_nsec = 1584200
     *          }
     *
     *          st_mtime   = {
     *              __tv_sec  = 1541445751
     *              __tv_nsec = 757015900
     *          }
     *
     *          st_ctime   = {
     *              __tv_sec  = 1541448277
     *              __tv_nsec = 534276400
     *          }
     *          st_blksize = 8192
     *          st_blocks  = 16
     *          st_fstype  = "tmpfs"
     *      }
     *
     */

    /* access time */
    printf ("ctime(&status_buffer.st_atime) = \"%s\"\n"
                                      , ctime(&status_buffer.st_atime));

    /* modification time */
    printf ("                    .st_mtime) = \"%s\"\n",
                                        ctime(&status_buffer.st_mtime));

    /* creation time */
    printf ("                    .st_ctime) = \"%s\"\n",
                                        ctime(&status_buffer.st_ctime));

    return EXIT_SUCCESS;

}

