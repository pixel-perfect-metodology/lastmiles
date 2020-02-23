	.arch armv7-a
	.eabi_attribute 28, 1
	.eabi_attribute 20, 1
	.eabi_attribute 21, 1
	.eabi_attribute 23, 3
	.eabi_attribute 24, 1
	.eabi_attribute 25, 1
	.eabi_attribute 26, 2
	.eabi_attribute 30, 6
	.eabi_attribute 34, 1
	.eabi_attribute 18, 4
	.file	"mbrot.c"
	.text
.Ltext0:
	.cfi_sections	.debug_frame
	.align	2
	.global	mbrot
	.syntax unified
	.arm
	.fpu vfpv4-d16
	.type	mbrot, %function
mbrot:
.LFB0:
	.file 1 "mbrot.c"
	.loc 1 19 0
	.cfi_startproc
	@ args = 0, pretend = 0, frame = 72
	@ frame_needed = 1, uses_anonymous_args = 0
	@ link register save eliminated.
	str	fp, [sp, #-4]!
	.cfi_def_cfa_offset 4
	.cfi_offset 11, -4
	add	fp, sp, #0
	.cfi_def_cfa_register 11
	sub	sp, sp, #76
	vstr.64	d0, [fp, #-60]
	vstr.64	d1, [fp, #-68]
	str	r0, [fp, #-72]
	.loc 1 23 0
	mov	r3, #0
	str	r3, [fp, #-8]
	.loc 1 24 0
	mov	r2, #0
	mov	r3, #0
	strd	r2, [fp, #-20]
	.loc 1 25 0
	mov	r2, #0
	mov	r3, #0
	strd	r2, [fp, #-28]
	.loc 1 27 0
	mov	r2, #0
	mov	r3, #0
	strd	r2, [fp, #-36]
	.loc 1 29 0
	b	.L2
.L4:
	.loc 1 30 0
	vldr.64	d6, [fp, #-20]
	vldr.64	d7, [fp, #-20]
	vmul.f64	d6, d6, d7
	vldr.64	d5, [fp, #-28]
	vldr.64	d7, [fp, #-28]
	vmul.f64	d7, d5, d7
	vsub.f64	d7, d6, d7
	vstr.64	d7, [fp, #-44]
	.loc 1 31 0
	vldr.64	d6, [fp, #-20]
	vldr.64	d7, [fp, #-28]
	vmul.f64	d7, d6, d7
	vadd.f64	d7, d7, d7
	vstr.64	d7, [fp, #-52]
	.loc 1 32 0
	vldr.64	d6, [fp, #-44]
	vldr.64	d7, [fp, #-60]
	vadd.f64	d7, d6, d7
	vstr.64	d7, [fp, #-20]
	.loc 1 33 0
	vldr.64	d6, [fp, #-52]
	vldr.64	d7, [fp, #-68]
	vadd.f64	d7, d6, d7
	vstr.64	d7, [fp, #-28]
	.loc 1 42 0
	vldr.64	d6, [fp, #-20]
	vldr.64	d7, [fp, #-20]
	vmul.f64	d6, d6, d7
	vldr.64	d5, [fp, #-28]
	vldr.64	d7, [fp, #-28]
	vmul.f64	d7, d5, d7
	vadd.f64	d7, d6, d7
	vstr.64	d7, [fp, #-36]
	.loc 1 44 0
	ldr	r3, [fp, #-8]
	add	r3, r3, #1
	str	r3, [fp, #-8]
.L2:
	.loc 1 29 0
	ldr	r2, [fp, #-8]
	ldr	r3, [fp, #-72]
	cmp	r2, r3
	bcs	.L3
	.loc 1 29 0 is_stmt 0 discriminator 1
	vldr.64	d7, [fp, #-36]
	vmov.f64	d6, #4.0e+0
	vcmpe.f64	d7, d6
	vmrs	APSR_nzcv, FPSCR
	bmi	.L4
.L3:
	.loc 1 47 0 is_stmt 1
	ldr	r3, [fp, #-8]
	.loc 1 49 0
	mov	r0, r3
	add	sp, fp, #0
	.cfi_def_cfa_register 13
	@ sp needed
	ldr	fp, [sp], #4
	.cfi_restore 11
	.cfi_def_cfa_offset 0
	bx	lr
	.cfi_endproc
.LFE0:
	.size	mbrot, .-mbrot
.Letext0:
	.file 2 "/usr/include/stdint.h"
	.section	.debug_info,"",%progbits
.Ldebug_info0:
	.4byte	0x120
	.2byte	0x4
	.4byte	.Ldebug_abbrev0
	.byte	0x4
	.uleb128 0x1
	.4byte	.LASF16
	.byte	0xc
	.4byte	.LASF17
	.4byte	.LASF18
	.4byte	.Ltext0
	.4byte	.Letext0-.Ltext0
	.4byte	.Ldebug_line0
	.uleb128 0x2
	.byte	0x1
	.byte	0x6
	.4byte	.LASF0
	.uleb128 0x2
	.byte	0x2
	.byte	0x5
	.4byte	.LASF1
	.uleb128 0x3
	.byte	0x4
	.byte	0x5
	.ascii	"int\000"
	.uleb128 0x2
	.byte	0x8
	.byte	0x5
	.4byte	.LASF2
	.uleb128 0x2
	.byte	0x1
	.byte	0x8
	.4byte	.LASF3
	.uleb128 0x2
	.byte	0x2
	.byte	0x7
	.4byte	.LASF4
	.uleb128 0x4
	.4byte	.LASF19
	.byte	0x2
	.byte	0x33
	.4byte	0x5a
	.uleb128 0x2
	.byte	0x4
	.byte	0x7
	.4byte	.LASF5
	.uleb128 0x2
	.byte	0x8
	.byte	0x7
	.4byte	.LASF6
	.uleb128 0x2
	.byte	0x4
	.byte	0x5
	.4byte	.LASF7
	.uleb128 0x2
	.byte	0x4
	.byte	0x7
	.4byte	.LASF8
	.uleb128 0x2
	.byte	0x4
	.byte	0x7
	.4byte	.LASF9
	.uleb128 0x2
	.byte	0x1
	.byte	0x8
	.4byte	.LASF10
	.uleb128 0x5
	.4byte	.LASF20
	.byte	0x1
	.byte	0x12
	.4byte	0x4f
	.4byte	.LFB0
	.4byte	.LFE0-.LFB0
	.uleb128 0x1
	.byte	0x9c
	.4byte	0x11c
	.uleb128 0x6
	.ascii	"c_r\000"
	.byte	0x1
	.byte	0x12
	.4byte	0x11c
	.uleb128 0x2
	.byte	0x91
	.sleb128 -64
	.uleb128 0x6
	.ascii	"c_i\000"
	.byte	0x1
	.byte	0x12
	.4byte	0x11c
	.uleb128 0x3
	.byte	0x91
	.sleb128 -72
	.uleb128 0x7
	.4byte	.LASF11
	.byte	0x1
	.byte	0x12
	.4byte	0x4f
	.uleb128 0x3
	.byte	0x91
	.sleb128 -76
	.uleb128 0x8
	.4byte	.LASF12
	.byte	0x1
	.byte	0x17
	.4byte	0x4f
	.uleb128 0x2
	.byte	0x91
	.sleb128 -12
	.uleb128 0x9
	.ascii	"zr\000"
	.byte	0x1
	.byte	0x18
	.4byte	0x11c
	.uleb128 0x2
	.byte	0x91
	.sleb128 -24
	.uleb128 0x9
	.ascii	"zi\000"
	.byte	0x1
	.byte	0x19
	.4byte	0x11c
	.uleb128 0x2
	.byte	0x91
	.sleb128 -32
	.uleb128 0x8
	.4byte	.LASF13
	.byte	0x1
	.byte	0x1a
	.4byte	0x11c
	.uleb128 0x2
	.byte	0x91
	.sleb128 -48
	.uleb128 0x8
	.4byte	.LASF14
	.byte	0x1
	.byte	0x1a
	.4byte	0x11c
	.uleb128 0x2
	.byte	0x91
	.sleb128 -56
	.uleb128 0x9
	.ascii	"mag\000"
	.byte	0x1
	.byte	0x1b
	.4byte	0x11c
	.uleb128 0x2
	.byte	0x91
	.sleb128 -40
	.byte	0
	.uleb128 0x2
	.byte	0x8
	.byte	0x4
	.4byte	.LASF15
	.byte	0
	.section	.debug_abbrev,"",%progbits
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
	.uleb128 0x6
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
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x5
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
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x6
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x2117
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x6
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x7
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x8
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x9
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_aranges,"",%progbits
	.4byte	0x1c
	.2byte	0x2
	.4byte	.Ldebug_info0
	.byte	0x4
	.byte	0
	.2byte	0
	.2byte	0
	.4byte	.Ltext0
	.4byte	.Letext0-.Ltext0
	.4byte	0
	.4byte	0
	.section	.debug_line,"",%progbits
.Ldebug_line0:
	.section	.debug_str,"MS",%progbits,1
.LASF2:
	.ascii	"long long int\000"
.LASF5:
	.ascii	"unsigned int\000"
.LASF14:
	.ascii	"tmp_i\000"
.LASF18:
	.ascii	"/home/dclarke/pgm/lastmiles/xmand\000"
.LASF13:
	.ascii	"tmp_r\000"
.LASF8:
	.ascii	"long unsigned int\000"
.LASF20:
	.ascii	"mbrot\000"
.LASF6:
	.ascii	"long long unsigned int\000"
.LASF16:
	.ascii	"GNU C99 6.3.0 20170516 -march=armv7-a -mtune=cortex"
	.ascii	"-a17 -mfpu=vfpv4-d16 -mstructure-size-boundary=32 -"
	.ascii	"marm -mtls-dialect=gnu2 -mno-sched-prolog -mlittle-"
	.ascii	"endian -mfloat-abi=hard -g -std=c99 -fno-builtin\000"
.LASF3:
	.ascii	"unsigned char\000"
.LASF10:
	.ascii	"char\000"
.LASF12:
	.ascii	"height\000"
.LASF19:
	.ascii	"uint32_t\000"
.LASF7:
	.ascii	"long int\000"
.LASF15:
	.ascii	"double\000"
.LASF4:
	.ascii	"short unsigned int\000"
.LASF0:
	.ascii	"signed char\000"
.LASF17:
	.ascii	"mbrot.c\000"
.LASF11:
	.ascii	"bail_out\000"
.LASF1:
	.ascii	"short int\000"
.LASF9:
	.ascii	"sizetype\000"
	.ident	"GCC: (Debian 6.3.0-18+deb9u1) 6.3.0 20170516"
	.section	.note.GNU-stack,"",%progbits
