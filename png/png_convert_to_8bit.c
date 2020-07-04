
#define _XOPEN_SOURCE 600

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

