// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSENUMS_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSENUMS_TD

include "mlir/IR/EnumAttr.td"

def TTNN_Reader : I32EnumAttrCase<"Reader", 0, "reader">;
def TTNN_Writer : I32EnumAttrCase<"Writer", 1, "writer">;

def TTNN_DataMovementType : I32EnumAttr<"DataMovementType", "TTNN DataMovementType",
                           [
                            TTNN_Reader,
                            TTNN_Writer,
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_LoFi : I32EnumAttrCase<"LoFi", 0, "lofi">;
def TTNN_HiFi2 : I32EnumAttrCase<"HiFi2", 1, "hifi2">;
def TTNN_HiFi3 : I32EnumAttrCase<"HiFi3", 2, "hifi3">;
def TTNN_HiFi4 : I32EnumAttrCase<"HiFi4", 3, "hifi4">;
def TTNN_Invalid : I32EnumAttrCase<"Invalid", 4, "invalid">;

def TTNN_MathFidelity : I32EnumAttr<"MathFidelity", "TTNN MathFidelity",
                           [
                            TTNN_LoFi,
                            TTNN_HiFi2,
                            TTNN_HiFi3,
                            TTNN_HiFi4,
                            TTNN_Invalid
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}

def TTNN_CB_c_in0 : I32EnumAttrCase<"c_in0", 0, "c_in0">;
def TTNN_CB_c_in1 : I32EnumAttrCase<"c_in1", 1, "c_in1">;
def TTNN_CB_c_in2 : I32EnumAttrCase<"c_in2", 2, "c_in2">;
def TTNN_CB_c_in3 : I32EnumAttrCase<"c_in3", 3, "c_in3">;
def TTNN_CB_c_in4 : I32EnumAttrCase<"c_in4", 4, "c_in4">;
def TTNN_CB_c_in5 : I32EnumAttrCase<"c_in5", 5, "c_in5">;
def TTNN_CB_c_in6 : I32EnumAttrCase<"c_in6", 6, "c_in6">;
def TTNN_CB_c_in7 : I32EnumAttrCase<"c_in7", 7, "c_in7">;

def TTNN_CB_dataflow0 : I32EnumAttrCase<"dataflow0", 8, "dataflow0">;
def TTNN_CB_dataflow1 : I32EnumAttrCase<"dataflow1", 9, "dataflow1">;
def TTNN_CB_dataflow2 : I32EnumAttrCase<"dataflow2", 10, "dataflow2">;
def TTNN_CB_dataflow3 : I32EnumAttrCase<"dataflow3", 11, "dataflow3">;
def TTNN_CB_dataflow4 : I32EnumAttrCase<"dataflow4", 12, "dataflow4">;
def TTNN_CB_dataflow5 : I32EnumAttrCase<"dataflow5", 13, "dataflow5">;
def TTNN_CB_dataflow6 : I32EnumAttrCase<"dataflow6", 14, "dataflow6">;
def TTNN_CB_dataflow7 : I32EnumAttrCase<"dataflow7", 15, "dataflow7">;

def TTNN_CB_c_out0 : I32EnumAttrCase<"c_out0", 16, "c_out0">;
def TTNN_CB_c_out1 : I32EnumAttrCase<"c_out1", 17, "c_out1">;
def TTNN_CB_c_out2 : I32EnumAttrCase<"c_out2", 18, "c_out2">;
def TTNN_CB_c_out3 : I32EnumAttrCase<"c_out3", 19, "c_out3">;
def TTNN_CB_c_out4 : I32EnumAttrCase<"c_out4", 20, "c_out4">;
def TTNN_CB_c_out5 : I32EnumAttrCase<"c_out5", 21, "c_out5">;
def TTNN_CB_c_out6 : I32EnumAttrCase<"c_out6", 22, "c_out6">;
def TTNN_CB_c_out7 : I32EnumAttrCase<"c_out7", 23, "c_out7">;

def TTNN_CB_c_intermed0 : I32EnumAttrCase<"c_intermed0", 24, "c_intermed0">;
def TTNN_CB_c_intermed1 : I32EnumAttrCase<"c_intermed1", 25, "c_intermed1">;
def TTNN_CB_c_intermed2 : I32EnumAttrCase<"c_intermed2", 26, "c_intermed2">;
def TTNN_CB_c_intermed3 : I32EnumAttrCase<"c_intermed3", 27, "c_intermed3">;
def TTNN_CB_c_intermed4 : I32EnumAttrCase<"c_intermed4", 28, "c_intermed4">;
def TTNN_CB_c_intermed5 : I32EnumAttrCase<"c_intermed5", 29, "c_intermed5">;
def TTNN_CB_c_intermed6 : I32EnumAttrCase<"c_intermed6", 30, "c_intermed6">;
def TTNN_CB_c_intermed7 : I32EnumAttrCase<"c_intermed7", 31, "c_intermed7">;

def TTNN_CB : I32EnumAttr<"CB", "CB Enumeration",
                           [
                            TTNN_CB_c_in0,
                            TTNN_CB_c_in1,
                            TTNN_CB_c_in2,
                            TTNN_CB_c_in3,
                            TTNN_CB_c_in4,
                            TTNN_CB_c_in5,
                            TTNN_CB_c_in6,
                            TTNN_CB_c_in7,
                            TTNN_CB_dataflow0,
                            TTNN_CB_dataflow1,
                            TTNN_CB_dataflow2,
                            TTNN_CB_dataflow3,
                            TTNN_CB_dataflow4,
                            TTNN_CB_dataflow5,
                            TTNN_CB_dataflow6,
                            TTNN_CB_dataflow7,
                            TTNN_CB_c_out0,
                            TTNN_CB_c_out1,
                            TTNN_CB_c_out2,
                            TTNN_CB_c_out3,
                            TTNN_CB_c_out4,
                            TTNN_CB_c_out5,
                            TTNN_CB_c_out6,
                            TTNN_CB_c_out7,
                            TTNN_CB_c_intermed0,
                            TTNN_CB_c_intermed1,
                            TTNN_CB_c_intermed2,
                            TTNN_CB_c_intermed3,
                            TTNN_CB_c_intermed4,
                            TTNN_CB_c_intermed5,
                            TTNN_CB_c_intermed6,
                            TTNN_CB_c_intermed7
                           ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::tt::ttnn";
}


#endif
