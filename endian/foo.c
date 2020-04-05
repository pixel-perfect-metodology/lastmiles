

#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdint.h>

int main(int argc,char *argv[])
{
    
    uint32_t a=1024;
    uint8_t bytes[4];
    uint8_t *p=(uint8_t*)&a;
    bytes[0]=p[0];
    bytes[1]=p[1];
    bytes[2]=p[2];
    bytes[3]=p[3];

    uint8_t bmp_header[12] = { 0, 1, 2, 3,
                               0, 4, 0, 0,
                               0,12,34, 1 };
    
    /* also for giggles 0x01df 0x5e76 */
    uint8_t not_pi[4] = { 0x01, 0xdf, 0x5e, 0x76 };
    uint8_t really_not_pi[4] = { 0x76, 0x5e, 0xdf, 0x01 };

    printf("%u\n",(bmp_header[7]<<24)|(bmp_header[6]<<16)|(bmp_header[5]<<8)|bmp_header[4]);


    printf("%u\n",(not_pi[3]<<24)|(not_pi[2]<<16)|(not_pi[1]<<8)|not_pi[0]);
    printf("%u\n",(really_not_pi[3]<<24)|(really_not_pi[2]<<16)|(really_not_pi[1]<<8)|really_not_pi[0]);

    printf("%u\n",(not_pi[0]<<24)|(not_pi[1]<<16)|(not_pi[2]<<8)|not_pi[3]);
    printf("%u\n",(really_not_pi[0]<<24)|(really_not_pi[1]<<16)|(really_not_pi[2]<<8)|really_not_pi[3]);

    return 0;
}

