	.file	"ieee754_ld.c"
	.abiversion 2
	.section	".text"
.Ltext0:
	.section	.rodata
	.align 3
.LC1:
	.string	"%02x "
	.align 3
.LC3:
	.string	"\n"
	.section	".toc","aw"
	.align 3
.LCTOC0:
	.tc .LCTOC1[TC],.LCTOC1
	.section	".toc1","aw"
	.align 3
.LCTOC1 = .+32768
.LC0:
	.quad	0x400921fb54442d18,0x3ca1a62633145c06
.LC2:
	.quad	.LC1
.LC4:
	.quad	.LC3
	.section	".text"
	.align 2
	.globl main
	.type	main, @function
main:
.LFB0:
	.file 1 "ieee754_ld.c"
	.loc 1 15 37
	.cfi_startproc
.LCF0:
0:	addis 2,12,.TOC.-.LCF0@ha
	addi 2,2,.TOC.-.LCF0@l
	.localentry	main,.-main
	mflr %r0
	std %r0,16(%r1)
	std %r30,-16(%r1)
	std %r31,-8(%r1)
	stdu %r1,-160(%r1)
	.cfi_def_cfa_offset 160
	.cfi_offset 65, 16
	.cfi_offset 30, -16
	.cfi_offset 31, -8
	mr %r31,%r1
	.cfi_def_cfa_register 31
	ld %r30,.LCTOC0@toc(%r2)
	mr %r9,%r3
	std %r4,136(%r31)
	stw %r9,128(%r31)
	.loc 1 18 17
	lfd %f0,.LC0-.LCTOC1(%r30)
	lfd %f1,.LC0+8-.LCTOC1(%r30)
	addi %r9,%r31,112
	stfd %f0,0(%r9)
	stfd %f1,8(%r9)
	.loc 1 22 12
	li %r9,0
	stw %r9,96(%r31)
	.loc 1 22 5
	b .L2
.L3:
	.loc 1 23 47 discriminator 3
	lwa %r9,96(%r31)
	addi %r10,%r31,112
	add %r9,%r10,%r9
	lbz %r9,0(%r9)
	.loc 1 23 9 discriminator 3
	extsw %r9,%r9
	mr %r4,%r9
	ld %r3,.LC2-.LCTOC1(%r30)
	bl printf
	nop
	.loc 1 22 40 discriminator 3
	lwz %r9,96(%r31)
	addi %r9,%r9,1
	stw %r9,96(%r31)
.L2:
	.loc 1 22 17 discriminator 1
	lwz %r9,96(%r31)
	.loc 1 22 5 discriminator 1
	cmplwi %cr0,%r9,15
	ble %cr0,.L3
	.loc 1 25 5
	ld %r3,.LC4-.LCTOC1(%r30)
	bl printf
	nop
	.loc 1 27 12
	li %r9,0
	.loc 1 28 1
	mr %r3,%r9
	addi %r1,%r31,160
	.cfi_def_cfa 1, 0
	ld %r0,16(%r1)
	mtlr %r0
	ld %r30,-16(%r1)
	ld %r31,-8(%r1)
	blr
	.long 0
	.byte 0,0,0,1,128,2,0,1
	.cfi_endproc
.LFE0:
	.size	main,.-main
.Letext0:
	.file 2 "/usr/include/powerpc64le-linux-gnu/bits/types.h"
	.file 3 "/usr/lib/gcc/powerpc64le-linux-gnu/9/include/stddef.h"
	.file 4 "/usr/include/powerpc64le-linux-gnu/bits/types/struct_FILE.h"
	.file 5 "/usr/include/powerpc64le-linux-gnu/bits/types/FILE.h"
	.file 6 "/usr/include/stdio.h"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.4byte	0x31e
	.2byte	0x4
	.4byte	.Ldebug_abbrev0
	.byte	0x8
	.uleb128 0x1
	.4byte	.LASF52
	.byte	0xc
	.4byte	.LASF53
	.4byte	.LASF54
	.8byte	.Ltext0
	.8byte	.Letext0-.Ltext0
	.4byte	.Ldebug_line0
	.uleb128 0x2
	.byte	0x1
	.byte	0x8
	.4byte	.LASF0
	.uleb128 0x2
	.byte	0x2
	.byte	0x7
	.4byte	.LASF1
	.uleb128 0x2
	.byte	0x4
	.byte	0x7
	.4byte	.LASF2
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.4byte	.LASF3
	.uleb128 0x2
	.byte	0x1
	.byte	0x6
	.4byte	.LASF4
	.uleb128 0x2
	.byte	0x2
	.byte	0x5
	.4byte	.LASF5
	.uleb128 0x3
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.4byte	.LASF6
	.uleb128 0x4
	.4byte	.LASF7
	.byte	0x2
	.byte	0x98
	.byte	0x19
	.4byte	0x5e
	.uleb128 0x4
	.4byte	.LASF8
	.byte	0x2
	.byte	0x99
	.byte	0x1b
	.4byte	0x5e
	.uleb128 0x5
	.byte	0x8
	.uleb128 0x6
	.byte	0x8
	.4byte	0x85
	.uleb128 0x2
	.byte	0x1
	.byte	0x8
	.4byte	.LASF9
	.uleb128 0x4
	.4byte	.LASF10
	.byte	0x3
	.byte	0xd1
	.byte	0x17
	.4byte	0x42
	.uleb128 0x7
	.4byte	.LASF55
	.byte	0xd8
	.byte	0x4
	.byte	0x31
	.byte	0x8
	.4byte	0x21f
	.uleb128 0x8
	.4byte	.LASF11
	.byte	0x4
	.byte	0x33
	.byte	0x7
	.4byte	0x57
	.byte	0
	.uleb128 0x8
	.4byte	.LASF12
	.byte	0x4
	.byte	0x36
	.byte	0x9
	.4byte	0x7f
	.byte	0x8
	.uleb128 0x8
	.4byte	.LASF13
	.byte	0x4
	.byte	0x37
	.byte	0x9
	.4byte	0x7f
	.byte	0x10
	.uleb128 0x8
	.4byte	.LASF14
	.byte	0x4
	.byte	0x38
	.byte	0x9
	.4byte	0x7f
	.byte	0x18
	.uleb128 0x8
	.4byte	.LASF15
	.byte	0x4
	.byte	0x39
	.byte	0x9
	.4byte	0x7f
	.byte	0x20
	.uleb128 0x8
	.4byte	.LASF16
	.byte	0x4
	.byte	0x3a
	.byte	0x9
	.4byte	0x7f
	.byte	0x28
	.uleb128 0x8
	.4byte	.LASF17
	.byte	0x4
	.byte	0x3b
	.byte	0x9
	.4byte	0x7f
	.byte	0x30
	.uleb128 0x8
	.4byte	.LASF18
	.byte	0x4
	.byte	0x3c
	.byte	0x9
	.4byte	0x7f
	.byte	0x38
	.uleb128 0x8
	.4byte	.LASF19
	.byte	0x4
	.byte	0x3d
	.byte	0x9
	.4byte	0x7f
	.byte	0x40
	.uleb128 0x8
	.4byte	.LASF20
	.byte	0x4
	.byte	0x40
	.byte	0x9
	.4byte	0x7f
	.byte	0x48
	.uleb128 0x8
	.4byte	.LASF21
	.byte	0x4
	.byte	0x41
	.byte	0x9
	.4byte	0x7f
	.byte	0x50
	.uleb128 0x8
	.4byte	.LASF22
	.byte	0x4
	.byte	0x42
	.byte	0x9
	.4byte	0x7f
	.byte	0x58
	.uleb128 0x8
	.4byte	.LASF23
	.byte	0x4
	.byte	0x44
	.byte	0x16
	.4byte	0x238
	.byte	0x60
	.uleb128 0x8
	.4byte	.LASF24
	.byte	0x4
	.byte	0x46
	.byte	0x14
	.4byte	0x23e
	.byte	0x68
	.uleb128 0x8
	.4byte	.LASF25
	.byte	0x4
	.byte	0x48
	.byte	0x7
	.4byte	0x57
	.byte	0x70
	.uleb128 0x8
	.4byte	.LASF26
	.byte	0x4
	.byte	0x49
	.byte	0x7
	.4byte	0x57
	.byte	0x74
	.uleb128 0x8
	.4byte	.LASF27
	.byte	0x4
	.byte	0x4a
	.byte	0xb
	.4byte	0x65
	.byte	0x78
	.uleb128 0x8
	.4byte	.LASF28
	.byte	0x4
	.byte	0x4d
	.byte	0x12
	.4byte	0x34
	.byte	0x80
	.uleb128 0x8
	.4byte	.LASF29
	.byte	0x4
	.byte	0x4e
	.byte	0xf
	.4byte	0x49
	.byte	0x82
	.uleb128 0x8
	.4byte	.LASF30
	.byte	0x4
	.byte	0x4f
	.byte	0x8
	.4byte	0x244
	.byte	0x83
	.uleb128 0x8
	.4byte	.LASF31
	.byte	0x4
	.byte	0x51
	.byte	0xf
	.4byte	0x254
	.byte	0x88
	.uleb128 0x8
	.4byte	.LASF32
	.byte	0x4
	.byte	0x59
	.byte	0xd
	.4byte	0x71
	.byte	0x90
	.uleb128 0x8
	.4byte	.LASF33
	.byte	0x4
	.byte	0x5b
	.byte	0x17
	.4byte	0x25f
	.byte	0x98
	.uleb128 0x8
	.4byte	.LASF34
	.byte	0x4
	.byte	0x5c
	.byte	0x19
	.4byte	0x26a
	.byte	0xa0
	.uleb128 0x8
	.4byte	.LASF35
	.byte	0x4
	.byte	0x5d
	.byte	0x14
	.4byte	0x23e
	.byte	0xa8
	.uleb128 0x8
	.4byte	.LASF36
	.byte	0x4
	.byte	0x5e
	.byte	0x9
	.4byte	0x7d
	.byte	0xb0
	.uleb128 0x8
	.4byte	.LASF37
	.byte	0x4
	.byte	0x5f
	.byte	0xa
	.4byte	0x8c
	.byte	0xb8
	.uleb128 0x8
	.4byte	.LASF38
	.byte	0x4
	.byte	0x60
	.byte	0x7
	.4byte	0x57
	.byte	0xc0
	.uleb128 0x8
	.4byte	.LASF39
	.byte	0x4
	.byte	0x62
	.byte	0x8
	.4byte	0x270
	.byte	0xc4
	.byte	0
	.uleb128 0x4
	.4byte	.LASF40
	.byte	0x5
	.byte	0x7
	.byte	0x19
	.4byte	0x98
	.uleb128 0x9
	.4byte	.LASF56
	.byte	0x4
	.byte	0x2b
	.byte	0xe
	.uleb128 0xa
	.4byte	.LASF41
	.uleb128 0x6
	.byte	0x8
	.4byte	0x233
	.uleb128 0x6
	.byte	0x8
	.4byte	0x98
	.uleb128 0xb
	.4byte	0x85
	.4byte	0x254
	.uleb128 0xc
	.4byte	0x42
	.byte	0
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.4byte	0x22b
	.uleb128 0xa
	.4byte	.LASF42
	.uleb128 0x6
	.byte	0x8
	.4byte	0x25a
	.uleb128 0xa
	.4byte	.LASF43
	.uleb128 0x6
	.byte	0x8
	.4byte	0x265
	.uleb128 0xb
	.4byte	0x85
	.4byte	0x280
	.uleb128 0xc
	.4byte	0x42
	.byte	0x13
	.byte	0
	.uleb128 0xd
	.4byte	.LASF44
	.byte	0x6
	.byte	0x89
	.byte	0xe
	.4byte	0x28c
	.uleb128 0x6
	.byte	0x8
	.4byte	0x21f
	.uleb128 0xd
	.4byte	.LASF45
	.byte	0x6
	.byte	0x8a
	.byte	0xe
	.4byte	0x28c
	.uleb128 0xd
	.4byte	.LASF46
	.byte	0x6
	.byte	0x8b
	.byte	0xe
	.4byte	0x28c
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.4byte	.LASF47
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.4byte	.LASF48
	.uleb128 0xe
	.4byte	.LASF57
	.byte	0x1
	.byte	0xf
	.byte	0x5
	.4byte	0x57
	.8byte	.LFB0
	.8byte	.LFE0-.LFB0
	.uleb128 0x1
	.byte	0x9c
	.4byte	0x314
	.uleb128 0xf
	.4byte	.LASF49
	.byte	0x1
	.byte	0xf
	.byte	0x10
	.4byte	0x57
	.uleb128 0x2
	.byte	0x91
	.sleb128 -32
	.uleb128 0xf
	.4byte	.LASF50
	.byte	0x1
	.byte	0xf
	.byte	0x1c
	.4byte	0x314
	.uleb128 0x2
	.byte	0x91
	.sleb128 -24
	.uleb128 0x10
	.string	"j"
	.byte	0x1
	.byte	0x11
	.byte	0x9
	.4byte	0x57
	.uleb128 0x2
	.byte	0x91
	.sleb128 -64
	.uleb128 0x10
	.string	"pi"
	.byte	0x1
	.byte	0x12
	.byte	0x11
	.4byte	0x31a
	.uleb128 0x2
	.byte	0x91
	.sleb128 -48
	.byte	0
	.uleb128 0x6
	.byte	0x8
	.4byte	0x7f
	.uleb128 0x2
	.byte	0x10
	.byte	0x4
	.4byte	.LASF51
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0xe
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x1b
	.uleb128 0xe
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0
	.byte	0
	.uleb128 0x4
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x5
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x6
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x7
	.uleb128 0x13
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x8
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x9
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xa
	.uleb128 0x13
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3c
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0xb
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xc
	.uleb128 0x21
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xd
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0xe
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x2116
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xf
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x10
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_aranges,"",@progbits
	.4byte	0x2c
	.2byte	0x2
	.4byte	.Ldebug_info0
	.byte	0x8
	.byte	0
	.2byte	0
	.2byte	0
	.8byte	.Ltext0
	.8byte	.Letext0-.Ltext0
	.8byte	0
	.8byte	0
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF8:
	.string	"__off64_t"
.LASF13:
	.string	"_IO_read_end"
.LASF10:
	.string	"size_t"
.LASF55:
	.string	"_IO_FILE"
.LASF46:
	.string	"stderr"
.LASF19:
	.string	"_IO_buf_end"
.LASF52:
	.ascii	"GNU C99 9.3.0 -msecure-plt -m64 -mcpu=power9 -"
	.string	"mno-isel -mno-crypto -mno-htm -mno-quad-memory-atomic -mfloat128-hardware -mfull-toc -mno-multiple -mupdate -mno-avoid-indexed-addresses -mno-toc -mregnames -mno-recip -g -O0 -std=c99 -fno-builtin -fno-unsafe-math-optimizations -fasynchronous-unwind-tables"
.LASF12:
	.string	"_IO_read_ptr"
.LASF37:
	.string	"__pad5"
.LASF2:
	.string	"unsigned int"
.LASF32:
	.string	"_offset"
.LASF38:
	.string	"_mode"
.LASF24:
	.string	"_chain"
.LASF53:
	.string	"ieee754_ld.c"
.LASF20:
	.string	"_IO_save_base"
.LASF0:
	.string	"unsigned char"
.LASF36:
	.string	"_freeres_buf"
.LASF43:
	.string	"_IO_wide_data"
.LASF3:
	.string	"long unsigned int"
.LASF1:
	.string	"short unsigned int"
.LASF44:
	.string	"stdin"
.LASF22:
	.string	"_IO_save_end"
.LASF56:
	.string	"_IO_lock_t"
.LASF23:
	.string	"_markers"
.LASF35:
	.string	"_freeres_list"
.LASF57:
	.string	"main"
.LASF40:
	.string	"FILE"
.LASF26:
	.string	"_flags2"
.LASF27:
	.string	"_old_offset"
.LASF31:
	.string	"_lock"
.LASF41:
	.string	"_IO_marker"
.LASF28:
	.string	"_cur_column"
.LASF48:
	.string	"long long unsigned int"
.LASF42:
	.string	"_IO_codecvt"
.LASF25:
	.string	"_fileno"
.LASF18:
	.string	"_IO_buf_base"
.LASF29:
	.string	"_vtable_offset"
.LASF33:
	.string	"_codecvt"
.LASF49:
	.string	"argc"
.LASF7:
	.string	"__off_t"
.LASF39:
	.string	"_unused2"
.LASF47:
	.string	"long long int"
.LASF45:
	.string	"stdout"
.LASF9:
	.string	"char"
.LASF15:
	.string	"_IO_write_base"
.LASF17:
	.string	"_IO_write_end"
.LASF5:
	.string	"short int"
.LASF21:
	.string	"_IO_backup_base"
.LASF11:
	.string	"_flags"
.LASF50:
	.string	"argv"
.LASF34:
	.string	"_wide_data"
.LASF6:
	.string	"long int"
.LASF16:
	.string	"_IO_write_ptr"
.LASF51:
	.string	"long double"
.LASF4:
	.string	"signed char"
.LASF14:
	.string	"_IO_read_base"
.LASF54:
	.string	"/home/dclarke/pgm/lastmiles/floating_point"
.LASF30:
	.string	"_shortbuf"
	.ident	"GCC: (Debian 9.3.0-10) 9.3.0"
	.gnu_attribute 4, 5
	.section	.note.GNU-stack,"",@progbits
