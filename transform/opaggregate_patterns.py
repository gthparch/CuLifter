PATTERN_TABLE = {
    ("PACK64",): [
        {
            "in": [
                {
                    "opcodes": ["PACK64"],
                    "def": ["reg1"],
                    "use": ["reg2", "reg3"],
                },
            ],
            "out": [
                {
                    "opcodes": ["MOV64"],
                    "def": ["reg1"],
                    "use": ["reg3:reg2"],
                }
            ],
            "next": [("reg2", "reg3")],
        }
    ],

    ("ISETP", "ISETP"): [
        {
            "in": [
                {
                    "opcodes": ["ISETP", "op1", "U32", "AND"],
                    "def": ["pred2", "PT"],
                    "use": ["reg3", "imm4", "PT"],
                },
                {
                    "opcodes": ["ISETP", "op1", "AND", "EX"],
                    "def": ["pred5", "PT"],
                    "use": ["reg6", "imm7", "PT", "pred2"],
                },
            ],
            "out": [
                {
                    "opcodes": ["ISETP64", "op1", "AND"],
                    "def": ["pred5", "PT"],
                    "use": ["reg6:reg3", "imm7:imm4", "PT"],
                }
            ],
            "next": [("reg3", "reg6")],
        },
        {
            "in": [
                {
                    "opcodes": ["ISETP", "op1", "U32", "AND"],
                    "def": ["pred2", "PT"],
                    "use": ["reg3", "imm4", "PT"],
                },
                {
                    "opcodes": ["ISETP", "op1", "U32", "AND", "EX"],
                    "def": ["pred5", "PT"],
                    "use": ["reg6", "imm7", "PT", "pred2"],
                },
            ],
            "out": [
                {
                    "opcodes": ["ISETP64", "op1", "AND"],
                    "def": ["pred5", "PT"],
                    "use": ["reg6:reg3", "imm7:imm4", "PT"],
                }
            ],
            "next": [("reg3", "reg6")],
        }
    ],
    ("LDG",): [
        {
            "in": [
                {
                    "opcodes": ["LDG", "E", "64", "SYS"],
                    "def": ["reg1", "reg2"],
                    "use": ["reg3"],
                },
            ],
            "out": [
                {
                    "opcodes": ["LDG64", "E", "SYS"],
                    "def": ["reg2:reg1"],
                    "use": ["reg3"],
                }
            ],
            "next": [("reg1", "reg2")],
        }
    ],
    ("LEA", "LEA"): [
        {
            "in": [
                {
                    "opcodes": ["LEA"],
                    "def": ["reg1", "pred2"],
                    "use": ["reg3", "op4", "imm5"],
                },
                {
                    "opcodes": ["LEA", "HI", "X"],
                    "def": ["reg6"],
                    "use": ["reg3", "op7", "reg8", "imm5", "pred2"],
                },
            ],
            "out": [
                {
                    "opcodes": ["LEA64"],
                    "def": ["reg6:reg1"],
                    "use": ["reg8:reg3", "op7:op4", "imm5"],
                }
            ],
            "next": [("reg3", "reg8")],
        }
    ],
	    ("PHI", "PHI"): [
	        {
	            "in": [
	                {
	                    "opcodes": ["PHI"],
	                    "def": ["reg1"],
	                    "use": ["reg_low[*]"],
	                },
	                {
	                    "opcodes": ["PHI"],
	                    "def": ["reg2"],
	                    "use": ["reg_high[*]"],
	                },
	            ],
	            "out": [
	                {
	                    "opcodes": ["PHI"],
	                    "def": ["reg2:reg1"],
	                    "use": ["reg_high[*]:reg_low[*]"],
	                }
	            ],
	            "next": [("reg_low[*]", "reg_high[*]")],
	        }
	    ],
    ("UIMAD",): [
        {
            "in": [
                {
                    "opcodes": ["UIMAD", "WIDE"],
                    "def": ["reg1", "reg2"],
                    "use": ["reg3", "op4", "op5"],
                },
            ],
            "out": [
                {
                    "opcodes": ["IMAD64"],
                    "def": ["reg2:reg1"],
                    "use": ["reg3", "op4", "op5"],
                }
            ],
            "next": [("reg1", "reg2")],
        },
    ],
    ("IMAD",): [
        {
            "in": [
                {
                    "opcodes": ["IMAD", "WIDE"],
                    "def": ["reg1", "reg2"],
                    "use": ["reg3", "op4", "op5"],
                },
            ],
            "out": [
                {
                    "opcodes": ["IMAD64"],
                    "def": ["reg2:reg1"],
                    "use": ["reg3", "op4", "op5"],
                }
            ],
        },
    ],
    ("MOV", "MOV"): [
        {
            "in": [
                {
                    "opcodes": ["MOV"],
                    "def": ["reg1"],
                    "use": ["reg2"],
                },
                {
                    "opcodes": ["MOV"],
                    "def": ["reg3"],
                    "use": ["reg4"],
                },
            ],
            "out": [
                {
                    "opcodes": ["MOV64"],
                    "def": ["reg3:reg1"],
                    "use": ["reg4:reg2"],
                }
            ],
            "next": [("reg2", "reg4")],
        }
    ],
    ("MOV",): [
        {
            "in": [
                {
                    "opcodes": ["MOV"],
                    "def": ["reg1"],
                    "use": ["reg2"],
                },
            ],
            "defined": ["reg3:reg2"],
            "out": [
                {
                    "opcodes": ["MOV64"],
                    "def": ["reg3:reg1"],
                    "use": ["reg3:reg2"],
                }
            ],
        },
        # {
        #     "in": [
        #         {
        #             "opcodes": ["IMAD", "MOV", "U32"],
        #             "def": ["reg1"],
        #             "use": ["RZ", "RZ", "reg2"],
        #         },
        #     ],
        #     "defined": ["reg3:reg2"],
        #     "out": [
        #         {
        #             "opcodes": ["AND64"],
        #             "def": ["RZ:reg1"],
        #             "use": ["reg3:reg2", "0x0000ffff"],
        #         },
        #     ],
        # },
    ],
    ("SHF",): [
        {
            "in": [
                {
                    "opcodes": ["SHF", "R", "S32", "HI"],
                    "def": ["reg1"],
                    "use": ["RZ", "0x1f", "reg2"],
                }
            ],
            "defined": ["reg2:reg3"],
            "out": [
                {
                    "opcodes": ["SHR64"],
                    "def": ["reg1:reg2"],
                    "use": ["reg2:reg3", "0x20"],
                },
            ],
        },
        {
            "in": [
                {
                    "opcodes": ["SHF", "R", "S32", "HI"],
                    "def": ["reg1"],
                    "use": ["RZ", "0x1f", "reg2"],
                }
            ],
            # "defined": ["reg1:reg2"],
            "out": [
                {
                    "opcodes": ["CAST64"],
                    "def": ["reg1:reg2"],
                    "use": ["reg2"],
                }
            ],
        }
    ],
    ("IADD3", "IADD3"): [
        { # weird pattern observed from wyllie
            "in": [
                {
                    "opcodes": ["IADD3"],
                    "def": ["reg1", "pred2", "pred3"],
                    "use": ["RZ", "reg4", "reg5"],
                },
                {
                    "opcodes": ["IADD3", "X"],
                    "def": ["reg6"],
                    "use": ["reg7", "RZ", "RZ", "pred2", "pred3"],
                },
            ],
            "out": [
                {
                    "opcodes": ["IADD364"],
                    "def": ["reg6:reg1"],
                    "use": ["RZ", "reg7:reg4", "RZ:reg5"],
                }
            ],
            "next": [("reg4", "reg7"), ("reg5", "RZ")],
        },
        {
            "in": [
                {
                    "opcodes": ["IADD3"],
                    "def": ["reg1", "pred1", "pred2"],
                    "use": ["reg2", "op3", "reg4"],
                },
                {
                    "opcodes": ["IADD3", "X"],
                    "def": ["reg5"],
                    "use": ["reg6", "op7", "reg8", "pred1", "pred2"],
                },
            ],
            "out": [
                {
                    "opcodes": ["IADD364"],
                    "def": ["reg5:reg1"],
                    "use": ["reg6:reg2", "op7:op3", "reg8:reg4"],
                }
            ],
            "next": [("reg2", "reg6"), ("op3", "op7"), ("reg4", "reg8")],
        },
        {
            "in": [
                {
                    "opcodes": ["IADD3"],
                    "def": ["reg1", "pred2"],
                    "use": ["reg3", "imm4", "RZ"],
                },
                {
                    "opcodes": ["IADD3", "X"],
                    "def": ["reg6"],
                    "use": ["RZ", "reg8", "RZ", "pred2", "!PT"],
                },
            ],
            "out": [
                {
                    "opcodes": ["IADD364"],
                    "def": ["reg6:reg1"],
                    "use": ["reg8:reg3", "imm4", "RZ"],
                }
            ],
            "next": [("reg3", "reg8")],
        },
        {
            "in": [
                {
                    "opcodes": ["IADD3"],
                    "def": ["reg1", "pred1"],
                    "use": ["reg2", "op3", "reg4"],
                },
                {
                    "opcodes": ["IADD3", "X"],
                    "def": ["reg5"],
                    "use": ["reg6", "op7", "reg8", "pred1"],
                },
            ],
            "out": [
                {
                    "opcodes": ["IADD364"],
                    "def": ["reg5:reg1"],
                    "use": ["reg6:reg2", "op7:op3", "reg8:reg4"],
                }
            ],
            "next": [("reg2", "reg6"), ("op3", "op7"), ("reg4", "reg8")],
        },
    ],
    # IADD3 R7, P5 = R6, 0x4, RZ, IMAD.X R13 = RZ, RZ, R12, P5
    ("IADD3", "IMAD"): [
        {
            "in": [
                {
                    "opcodes": ["IADD3"],
                    "def": ["reg1", "pred2"],
                    "use": ["reg3", "imm4", "RZ"],
                },
                {
                    "opcodes": ["IMAD", "X"],
                    "def": ["reg5"],
                    "use": ["RZ", "RZ", "reg6", "pred2"],
                },
            ],
            "out": [
                {
                    "opcodes": ["IADD364"],
                    "def": ["reg5:reg1"],
                    "use": ["reg6:reg3", "imm4", "RZ"],
                }
            ],
            "next": [("reg3", "reg6")],
        },
    ],
    # IMAD.SHL.U32 R186 = R144, 0x4, RZ, SHF.L.U64.HI R209 = R144, 0x2, R151
    ("IMAD", "SHF"): [
        {
            "in": [
                {
                    "opcodes": ["IMAD", "SHL", "U32"],
                    "def": ["reg1"],
                    "use": ["reg2", "imm3", "RZ"],
                },
                {
                    "opcodes": ["SHF", "L", "U64", "HI"],
                    "def": ["reg4"],
                    "use": ["reg2", "imm5", "reg6"],
                },
            ],
            "out": [
                {
                    "opcodes": ["SHL64"],
                    "def": ["reg4:reg1"],
                    "use": ["reg6:reg2", "imm5", "RZ"],
                }
            ],
            "next": [("reg2", "reg6")],
        },
    ],
    ("SHF", "SHF"): [
        {
            "in": [
                {
                    "opcodes": ["SHF", "L", "U64", "HI"],
                    "def": ["reg1"],
                    "use": ["reg2", "imm3", "reg4"],
                },
                {
                    "opcodes": ["SHF", "L", "U32"],
                    "def": ["reg5"],
                    "use": ["reg2", "imm3", "RZ"],
                },
            ],
            "out": [
                {
                    "opcodes": ["SHL64"],
                    "def": ["reg1:reg5"],
                    "use": ["reg4:reg2", "imm3"],
                }
            ],
            "next": [("reg2", "reg4")],
        },
    ],
}
