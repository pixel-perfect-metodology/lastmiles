	.text
	.file	"ieee754_ld.c"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4               # -- Begin function main
.LCPI0_0:
	.quad	-3958705157555305931    # x86_fp80 3.14159265358979323851
	.short	16384
	.zero	6
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "/home/dclarke/pgm/lastmiles/floating_point" "ieee754_ld.c"
	.loc	1 15 0                  # ieee754_ld.c:15:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$64, %rsp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	fldt	.LCPI0_0(%rip)
.Ltmp0:
	.loc	1 18 17 prologue_end    # ieee754_ld.c:18:17
	fstpt	-48(%rbp)
.Ltmp1:
	.loc	1 22 12                 # ieee754_ld.c:22:12
	movl	$0, -20(%rbp)
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
.Ltmp2:
	.loc	1 22 16 is_stmt 0       # ieee754_ld.c:22:16
	movslq	-20(%rbp), %rax
	.loc	1 22 17                 # ieee754_ld.c:22:17
	cmpq	$16, %rax
.Ltmp3:
	.loc	1 22 5                  # ieee754_ld.c:22:5
	jae	.LBB0_4
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
.Ltmp4:
	.loc	1 23 25 is_stmt 1       # ieee754_ld.c:23:25
	movslq	-20(%rbp), %rax
	movzbl	-48(%rbp,%rax), %esi
	.loc	1 23 9 is_stmt 0        # ieee754_ld.c:23:9
	movabsq	$.L.str, %rdi
	movb	$0, %al
	callq	printf
	movl	%eax, -52(%rbp)         # 4-byte Spill
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	.loc	1 22 40 is_stmt 1       # ieee754_ld.c:22:40
	movl	-20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -20(%rbp)
	.loc	1 22 5 is_stmt 0        # ieee754_ld.c:22:5
	jmp	.LBB0_1
.Ltmp5:
.LBB0_4:
	.loc	1 25 5 is_stmt 1        # ieee754_ld.c:25:5
	movabsq	$.L.str.1, %rdi
	movb	$0, %al
	callq	printf
	xorl	%ecx, %ecx
	movl	%eax, -56(%rbp)         # 4-byte Spill
	.loc	1 27 5                  # ieee754_ld.c:27:5
	movl	%ecx, %eax
	addq	$64, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Ltmp6:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object          # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"%02x "
	.size	.L.str, 6

	.type	.L.str.1,@object        # @.str.1
.L.str.1:
	.asciz	"\n"
	.size	.L.str.1, 2

	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"FreeBSD clang version 8.0.1 (tags/RELEASE_801/final 366581) (based on LLVM 8.0.1)" # string offset=0
.Linfo_string1:
	.asciz	"ieee754_ld.c"          # string offset=82
.Linfo_string2:
	.asciz	"/home/dclarke/pgm/lastmiles/floating_point" # string offset=95
.Linfo_string3:
	.asciz	"unsigned char"         # string offset=138
.Linfo_string4:
	.asciz	"main"                  # string offset=152
.Linfo_string5:
	.asciz	"int"                   # string offset=157
.Linfo_string6:
	.asciz	"argc"                  # string offset=161
.Linfo_string7:
	.asciz	"argv"                  # string offset=166
.Linfo_string8:
	.asciz	"char"                  # string offset=171
.Linfo_string9:
	.asciz	"j"                     # string offset=176
.Linfo_string10:
	.asciz	"pi"                    # string offset=178
.Linfo_string11:
	.asciz	"long double"           # string offset=181
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	16                      # DW_AT_stmt_list
	.byte	6                       # DW_FORM_data4
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	1                       # DW_FORM_addr
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	15                      # DW_TAG_pointer_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	1                       # DW_FORM_addr
	.byte	64                      # DW_AT_frame_base
	.byte	10                      # DW_FORM_block1
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	39                      # DW_AT_prototyped
	.byte	12                      # DW_FORM_flag
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	63                      # DW_AT_external
	.byte	12                      # DW_FORM_flag
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	5                       # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	10                      # DW_FORM_block1
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	6                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	10                      # DW_FORM_block1
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	2                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0xa7 DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_producer
	.short	12                      # DW_AT_language
	.long	.Linfo_string1          # DW_AT_name
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.long	.Linfo_string2          # DW_AT_comp_dir
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.quad	.Lfunc_end0             # DW_AT_high_pc
	.byte	2                       # Abbrev [2] 0x2e:0x5 DW_TAG_pointer_type
	.long	51                      # DW_AT_type
	.byte	3                       # Abbrev [3] 0x33:0x7 DW_TAG_base_type
	.long	.Linfo_string3          # DW_AT_name
	.byte	8                       # DW_AT_encoding
	.byte	1                       # DW_AT_byte_size
	.byte	4                       # Abbrev [4] 0x3a:0x58 DW_TAG_subprogram
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.quad	.Lfunc_end0             # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	86
	.long	.Linfo_string4          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	15                      # DW_AT_decl_line
	.byte	1                       # DW_AT_prototyped
	.long	146                     # DW_AT_type
	.byte	1                       # DW_AT_external
	.byte	5                       # Abbrev [5] 0x59:0xe DW_TAG_formal_parameter
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	120
	.long	.Linfo_string6          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	15                      # DW_AT_decl_line
	.long	146                     # DW_AT_type
	.byte	5                       # Abbrev [5] 0x67:0xe DW_TAG_formal_parameter
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string7          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	15                      # DW_AT_decl_line
	.long	153                     # DW_AT_type
	.byte	6                       # Abbrev [6] 0x75:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	108
	.long	.Linfo_string9          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	17                      # DW_AT_decl_line
	.long	146                     # DW_AT_type
	.byte	6                       # Abbrev [6] 0x83:0xe DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	80
	.long	.Linfo_string10         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	18                      # DW_AT_decl_line
	.long	170                     # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	3                       # Abbrev [3] 0x92:0x7 DW_TAG_base_type
	.long	.Linfo_string5          # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	2                       # Abbrev [2] 0x99:0x5 DW_TAG_pointer_type
	.long	158                     # DW_AT_type
	.byte	2                       # Abbrev [2] 0x9e:0x5 DW_TAG_pointer_type
	.long	163                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0xa3:0x7 DW_TAG_base_type
	.long	.Linfo_string8          # DW_AT_name
	.byte	6                       # DW_AT_encoding
	.byte	1                       # DW_AT_byte_size
	.byte	3                       # Abbrev [3] 0xaa:0x7 DW_TAG_base_type
	.long	.Linfo_string11         # DW_AT_name
	.byte	4                       # DW_AT_encoding
	.byte	16                      # DW_AT_byte_size
	.byte	0                       # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_macinfo,"",@progbits
	.byte	0                       # End Of Macro List Mark

	.ident	"FreeBSD clang version 8.0.1 (tags/RELEASE_801/final 366581) (based on LLVM 8.0.1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym printf
	.section	.debug_line,"",@progbits
.Lline_table_start0:
