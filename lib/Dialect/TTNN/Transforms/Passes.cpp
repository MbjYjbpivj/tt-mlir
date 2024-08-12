// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNOPENDEVICE
#define GEN_PASS_DEF_TTNNGENERIC
#define GEN_PASS_DEF_CONVERTTTIRTOTTNN
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNOpenDevice : public impl::TTNNOpenDeviceBase<TTNNOpenDevice> {
public:
  using impl::TTNNOpenDeviceBase<TTNNOpenDevice>::TTNNOpenDeviceBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module);
    auto systemDesc = llvm::cast<tt::SystemDescAttr>(
        module->getAttr(tt::SystemDescAttr::name));
    auto chipDescIndices = systemDesc.getChipDescIndices();
    assert(chipDescIndices.size() == 1 && "Multiple chips not supported yet");

    module->walk([&](func::FuncOp func) {
      // For now just push the open and close device ops to the beginning and
      // end of the function
      assert(func.getBody().hasOneBlock());
      auto *block = &func.getBody().front();
      auto opRange = block->without_terminator();

      builder.setInsertionPoint(block, opRange.begin());
      auto openDevice = builder.create<OpenDeviceOp>(
          func.getLoc(), builder.getType<tt::DeviceType>(
                             builder.getAttr<tt::DeviceAttr>(systemDesc)));

      builder.setInsertionPoint(block, opRange.end());
      builder.create<CloseDeviceOp>(func.getLoc(), openDevice.getResult());
    });
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
  }
};

// Rewrites `ttnn.add` or `ttnn.multiply` call to `ttnn.kernel`.
template <typename TTNNOpType>
class TTNNNamedTTNNOpToKernelOpRewriter : public OpRewritePattern<TTNNOpType> {
public:
  using OpRewritePattern<TTNNOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTNNOpType op,
                                PatternRewriter &rewriter) const final {
    StringRef kernelName;
    StringRef kernelKind;

    if constexpr (std::is_same<TTNNOpType, ttnn::MultiplyOp>::value) {
      kernelName = "mulitply";
      kernelKind = "eltwise";
    } else if constexpr (std::is_same<TTNNOpType, ttnn::AddOp>::value) {
      kernelName = "add";
      kernelKind = "eltwise";
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported Tosa operation for TTNN");
    }

    assert(kernelName.size() > 0);

    auto kernel = rewriter.create<ttnn::KernelOp>(
        op.getLoc(), op.getResultTypes(), kernelName, kernelKind,
        op.getInputs(), op.getOutputs());

    rewriter.replaceOp(op, kernel);

    return success();
  }
};

class TTNNKernelOpToGenericOpRewriter : public OpRewritePattern<KernelOp> {
private:
  SmallVector<Attribute>
  create_circular_buffer_attributes(PatternRewriter &rewriter,
                                    CoreRangeAttr &core_range) const {
    auto data_format = DataType::BFloat16;
    auto tile = rewriter.getAttr<TileType>(32, 32, data_format);
    auto page_size = tile.getSizeBytes();
    auto total_size = 2 * page_size;

    auto in0_circular_buffer_attributes =
        rewriter.getAttr<CircularBufferAttributesAttr>(
            CB::c_in0, core_range, total_size, page_size, data_format);

    auto in1_circular_buffer_attributes =
        rewriter.getAttr<CircularBufferAttributesAttr>(
            CB::c_in1, core_range, total_size, page_size, data_format);

    auto out0_circular_buffer_attributes =
        rewriter.getAttr<CircularBufferAttributesAttr>(
            CB::c_out0, core_range, total_size, page_size, data_format);

    return {in0_circular_buffer_attributes, in1_circular_buffer_attributes,
            out0_circular_buffer_attributes};
  }

  SmallVector<Attribute>
  create_data_movement_attributes(PatternRewriter &rewriter,
                                  CoreRangeAttr &core_range) const {
    const char *reader_kernel_path =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/"
        "dataflow/reader_binary_interleaved_start_id.cpp";

    // Assuming interleaved mem layout where inputs are in DRAM.
    auto src0_is_dram = true;
    auto src1_is_dram = true;
    SmallVector<uint32_t> reader_compile_time_args = {src0_is_dram,
                                                      src1_is_dram};

    auto reader_config = rewriter.getAttr<DataMovementConfigAttr>(
        DataMovementType::Reader, reader_compile_time_args);

    auto reader_attributes = rewriter.getAttr<DataMovementAttributesAttr>(
        core_range, rewriter.getAttr<StringAttr>(reader_kernel_path),
        reader_config);

    const char *writer_kernel_path =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id.cpp";

    // Assuming interleaved mem layout where outputs are in DRAM.
    auto dst_is_dram = true;
    SmallVector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(CB::c_out0), dst_is_dram};

    auto writer_config = rewriter.getAttr<DataMovementConfigAttr>(
        DataMovementType::Reader, reader_compile_time_args);

    auto writer_attributes = rewriter.getAttr<DataMovementAttributesAttr>(
        core_range, rewriter.getAttr<StringAttr>(writer_kernel_path),
        writer_config);

    return {reader_attributes, writer_attributes};
  }

  mlir::tt::GridAttr get_grid(PatternRewriter &rewriter) const {
    return GridAttr::get(rewriter.getContext(), {6, 6});
  }

public:
  using OpRewritePattern<KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(KernelOp op,
                                PatternRewriter &rewriter) const final {
    // Test if this generic op has already been lowered, todo find a better way
    if (op.getOperation()->getParentOp()->getName() ==
        OperationName(GenericOp::getOperationName(), rewriter.getContext())) {
      return failure();
    }

    GridAttr grid = get_grid(rewriter);
    // Core range over the entire specified grid.
    auto all_cores = rewriter.getAttr<CoreRangeAttr>(grid);

    auto circular_buffer_attributes =
        create_circular_buffer_attributes(rewriter, all_cores);

    auto generic_op = rewriter.create<GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), grid,
        rewriter.getArrayAttr(circular_buffer_attributes));

    // Create a new basic block for the generic_op op and create block arguments
    Block *block = rewriter.createBlock(&generic_op.getRegion());

    SmallVector<Location> blockArgumentLocs(generic_op.getOperands().size(),
                                            generic_op.getLoc());

    block->addArguments(TypeRange(generic_op.getOperandTypes()),
                        blockArgumentLocs);

    // Update the operands of the original op to use the block arguments
    op.getOperation()->setOperands(block->getArguments());

    // Move the original op into the generic_op block
    Operation *operation = op.getOperation()->clone();
    block->push_back(operation);
    rewriter.setInsertionPoint(block, block->end());
    rewriter.create<ttnn::YieldOp>(generic_op.getLoc(),
                                   ValueRange({operation->getResult(0)}));
    rewriter.replaceOp(op, generic_op);
    return success();
  }
};

class TTNNGeneric : public impl::TTNNGenericBase<TTNNGeneric> {
public:
  using impl::TTNNGenericBase<TTNNGeneric>::TTNNGenericBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());

    patterns.add<TTNNNamedTTNNOpToKernelOpRewriter<AddOp>,
                 TTNNNamedTTNNOpToKernelOpRewriter<MultiplyOp>,
                 TTNNKernelOpToGenericOpRewriter>(&getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

} // namespace mlir::tt::ttnn
