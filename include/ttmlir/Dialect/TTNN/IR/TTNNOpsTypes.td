// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSTYPES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "ttmlir/Dialect/TT/IR/TTOpsEnums.td"
include "ttmlir/Dialect/TTNN/IR/TTNNBase.td"
include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.td"

//===----------------------------------------------------------------------===//
// TTNN type definitions
//===----------------------------------------------------------------------===//

class TTNN_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TTNN_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// TTNN attr definitions
//===----------------------------------------------------------------------===//
// Should Attr be a separate file?

class TTNN_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<TTNN_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "ttnn." # attrMnemonic;
}

def TTNN_CoreRangeAttr : TTNN_Attr<"CoreRange", "core_range"> {
  let summary = "TTNN grid attribute";
  let description = [{
    TTNN grid attribute
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$offset,
                        ArrayRefParameter<"int64_t">:$size);
  let assemblyFormat = "`<` custom<DimensionList>($offset) `,` custom<DimensionList>($size) `>`";

  let extraClassDeclaration = [{
      static CoreRangeAttr get(::mlir::MLIRContext *context, ::mlir::tt::GridAttr grid, SmallVector<int64_t> offset = {0, 0})
      {
        assert(grid.getShape().size() == 2 && "Grid shape must be 2D for now");
        return CoreRangeAttr::get(context, offset, grid.getShape());
      }
  }];
}

def TTNN_CoreRangeArrayAttr : TypedArrayAttrBase<TTNN_CoreRangeAttr, "">;

def TTNN_CircularBufferAttributesAttr : TTNN_Attr<"CircularBufferAttributes", "circular_buffer_attributes"> {
  let summary = "Circular buffer attributes";
  let description = [{
    An attribute representing the circular_buffer_attributes_t structure.
  }];

  let parameters = (ins 
    "CB":$cb_id,
    TTNN_CoreRangeAttr:$core_range,
    "uint32_t":$total_size,
    "uint32_t":$page_size,
    "DataType":$data_format
  );

  // Define the assembly format for the attribute
  let assemblyFormat = "`<` $cb_id `,` $core_range `,` $total_size `,` $page_size `,` $data_format `>`";
}

// An array of circular buffer attributes.
def TTNN_CircularBufferAttributesArrayAttr : TypedArrayAttrBase<TTNN_CircularBufferAttributesAttr, "">;

def TTNN_DataMovementConfigAttr : TTNN_Attr<"DataMovementConfig", "data_movement_config"> {
  let summary = "DataMovementConfig";
  let description = [{
    DataMovementConfig
  }];

  let parameters = (ins 
    "DataMovementType":$data_movement_type,
    ArrayRefParameter<"uint32_t">:$compile_args);

  let assemblyFormat = "`<` $data_movement_type `,` $compile_args `>`";
}

def TTNN_DataMovementAttributesAttr : TTNN_Attr<"DataMovementAttributes", "data_movement_attributes"> {
  let summary = "Data movement attributes";
  let description = [{
    An attribute representing the data_movement_attributes_t structure.
  }];

  let parameters = (ins 
    TTNN_CoreRangeAttr:$core_range,
    "::mlir::StringAttr":$kernel_path,
    TTNN_DataMovementConfigAttr:$data_movement_config);
    // TODO runtime_args missing

  let assemblyFormat = "`<` $core_range `,` $kernel_path `,` $data_movement_config `>`";
}

// An array of data movement attributes.
def TTNN_DataMovementAttributesArrayAttr : TypedArrayAttrBase<TTNN_DataMovementAttributesAttr, "">;

def TTNN_ComputeConfig : TTNN_Attr<"ComputeConfig", "compute_config"> {
  let summary = "ComputeConfig";
  let description = [{
    ComputeConfig
  }];

  let parameters = (ins 
    "MathFidelity":$math_fidelity,
    "::mlir::BoolAttr":$fp32_dest_acc_en,
    "::mlir::BoolAttr":$preserve_fp32_precision,
    "::mlir::BoolAttr":$math_approx_mode,
    ArrayRefParameter<"uint32_t">:$compile_args);

  let assemblyFormat = "`<` $math_fidelity `,` $fp32_dest_acc_en `,` $preserve_fp32_precision `,` $math_approx_mode `,` $compile_args `>`";
}

def TTNN_ComputeAttributesAttr : TTNN_Attr<"ComputeAttributes", "compute_attributes"> {
  let summary = "Data movement attributes";
  let description = [{
    An attribute representing the compute_attributes_t structure.
  }];

  let parameters = (ins 
    TTNN_CoreRangeAttr:$core_range,
    "::mlir::StringAttr":$kernel_path,
    TTNN_ComputeConfig:$compute_config);
    // TODO runtime_args missing

  let assemblyFormat = "`<` $core_range `,` $kernel_path `,` $compute_config `>`";
}

// An array of compute attributes.
def TTNN_ComputeAttributesArrayAttr : TypedArrayAttrBase<TTNN_ComputeAttributesAttr, "">;

// def TTNN_OperationAttributesAttr : TTNN_Attr<"OperationAttributes", "operation_attributes"> {
//   let summary = "Operation attributes";
//   let description = [{
//     An attribute representing the operation_attributest_t structure.
//   }];

//   let parameters = (ins 
//     TTNN_CircularBufferAttributesArrayAttr:$circular_buffer_attributes);
//     TTNN_DataMovementAttributesArrayAttr:$data_movement_attributes,
//     TTNN_ComputeAttributesArrayAttr:$compute_attributes);

//   let assemblyFormat = "`<` $circular_buffer_attributes `,` $data_movement_attributes `,` $compute_attributes `>`";
// }

#endif
