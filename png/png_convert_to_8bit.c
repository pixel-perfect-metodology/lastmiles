
/*
 * png_convert_to_8bit.c trivial sample of libpng
 * Maintained 2020 Dennis Clarke
 * Maintained 2018 Cosmin Truta
 * Maintained 1998-2016 Glenn Randers-Pehrson
 * Maintained 1996-1997 Andreas Dilger
 * Written 1995-1996 Guy Eric Schalnat, Group 42, Inc.
 *
 * To the extent possible under law, the authors have waived
 * all copyright and related or neighboring rights to this file.
 * This work is published from: United States, Canada.
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

/* This is an example of how to use libpng to read and write PNG files.
 * The file libpng-manual.txt is much more verbose then this.  If you have
 * not read it, do so first.  This was designed to be a starting point of an
 * implementation.  This is not officially part of libpng, is hereby placed
 * in the public domain, and therefore does not require a copyright notice.
 *
 * However, having said that we must at least provide code that works
 * in that is does some libpng operations correctly and delivers a 
 * valid result.
 *
 * The simple, but restricted approach to reading a PNG file or data stream
 * requires just two function calls, as in the following complete program.
 * Writing a file needs just one function call, so long as the data has an
 * appropriate layout.
 *
 * The following code reads PNG image data from a file and writes it, in a
 * potentially new format, to a new file.
 * 
 * This code provides little or no error checking. 
 * For a more realistic version see contrib/examples/pngtopng.c
 */

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <png.h>
#include <zlib.h>

int main(int argc, const char **argv) {
    png_image image;
    png_bytep buffer;
    if (argc == 3) {
        memset(&image, 0, (size_t)sizeof(image));
        image.version = PNG_IMAGE_VERSION;
        if (png_image_begin_read_from_file(&image, argv[1]) != 0) {
            image.format = PNG_FORMAT_RGBA;
            buffer = malloc(PNG_IMAGE_SIZE(image));
            if ( buffer == NULL ) {
                if ( errno == ENOMEM ) {
                    fprintf(stderr,"FAIL : calloc returns ENOMEM at %s:%d\n", __FILE__, __LINE__ );
                } else {
                    fprintf(stderr,"FAIL : calloc fails at %s:%d\n", __FILE__, __LINE__ );
                }
                perror("FAIL ");
                return EXIT_FAILURE;
            }
            if ( png_image_finish_read(&image,
                                       NULL /*background*/,
                                       buffer,
                                       0 /*row_stride*/,
                                       NULL/*colormap*/) != 0 ) {

                if ( png_image_write_to_file(&image, argv[2],
                                             0 /*convert_to_8bit*/,
                                             buffer,
                                             0 /*row_stride*/,
                                             NULL/*colormap*/) != 0 ) {

                    /* did we free the buffer ? */
                    png_image_free(&image);
                    free(buffer);
                    return 42;
                }
            }
        } else {
            fprintf(stderr, "FAIL : file %s is borked\n", argv[1]);
            return EXIT_FAILURE;
        }
    }
    fprintf(stderr,"FAIL : usage %s file1 file2\n", argv[0]);
    return EXIT_FAILURE;
}

