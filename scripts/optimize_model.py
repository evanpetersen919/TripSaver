"""
Model Performance Optimization
===============================

Optimizes trained models for production deployment:
- TensorRT conversion for NVIDIA GPUs
- ONNX export for cross-platform deployment
- Model pruning (structured & unstructured)
- Quantization (INT8 & FP16)
- Inference profiling and benchmarking

Author: Evan Petersen  
Date: November 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.landmark_detector import LandmarkDetector


# ============================================================================
# ONNX EXPORT
# ============================================================================

def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    save_path: str,
    opset_version: int = 14,
    dynamic_axes: Optional[Dict] = None
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        dummy_input: Example input tensor
        save_path: Path to save ONNX model
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable batch size
    """
    print(f"Exporting model to ONNX...")
    
    model.eval()
    
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"✓ Model exported to {save_path}")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")


# ============================================================================
# TENSORRT CONVERSION
# ============================================================================

def convert_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    precision: str = 'fp16',
    workspace_size: int = 1 << 30,  # 1GB
    max_batch_size: int = 32
):
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', or 'int8')
        workspace_size: Maximum workspace size in bytes
        max_batch_size: Maximum batch size
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: TensorRT not installed. Install with: pip install tensorrt")
        return
    
    print(f"Converting to TensorRT ({precision})...")
    
    # Create builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size
    
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 precision enabled")
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # TODO: Add INT8 calibrator
        print("✓ INT8 precision enabled")
    
    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    engine = builder.build_engine(network, config)
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"✓ TensorRT engine saved to {engine_path}")


# ============================================================================
# MODEL PRUNING
# ============================================================================

def prune_model(
    model: nn.Module,
    amount: float = 0.3,
    method: str = 'l1_unstructured'
) -> nn.Module:
    """
    Apply pruning to reduce model size.
    
    Args:
        model: PyTorch model
        amount: Fraction of parameters to prune (0-1)
        method: Pruning method ('l1_unstructured', 'l1_structured', 'random')
        
    Returns:
        Pruned model
    """
    import torch.nn.utils.prune as prune
    
    print(f"Pruning model ({method}, amount={amount})...")
    
    parameters_to_prune = []
    
    # Collect parameters
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply pruning
    if method == 'l1_unstructured':
        for module, param in parameters_to_prune:
            prune.l1_unstructured(module, name=param, amount=amount)
    elif method == 'random':
        for module, param in parameters_to_prune:
            prune.random_unstructured(module, name=param, amount=amount)
    
    # Make pruning permanent
    for module, param in parameters_to_prune:
        prune.remove(module, param)
    
    # Calculate sparsity
    total_params = 0
    zero_params = 0
    
    for module, param in parameters_to_prune:
        tensor = getattr(module, param)
        total_params += tensor.numel()
        zero_params += (tensor == 0).sum().item()
    
    sparsity = 100.0 * zero_params / total_params
    print(f"✓ Model pruned: {sparsity:.2f}% sparsity")
    
    return model


# ============================================================================
# QUANTIZATION
# ============================================================================

def quantize_model(
    model: nn.Module,
    calibration_loader: Optional[DataLoader] = None,
    backend: str = 'fbgemm'
) -> nn.Module:
    """
    Apply dynamic or static quantization.
    
    Args:
        model: PyTorch model
        calibration_loader: Data loader for calibration (static quantization)
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        
    Returns:
        Quantized model
    """
    print(f"Quantizing model (backend={backend})...")
    
    model.eval()
    
    if calibration_loader is not None:
        # Static quantization (requires calibration)
        print("Performing static quantization with calibration...")
        
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate
        with torch.no_grad():
            for images, _ in tqdm(calibration_loader, desc="Calibrating"):
                model_prepared(images)
        
        model_quantized = torch.quantization.convert(model_prepared)
        print("✓ Static quantization complete")
    
    else:
        # Dynamic quantization (no calibration needed)
        print("Performing dynamic quantization...")
        
        model_quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        print("✓ Dynamic quantization complete")
    
    return model_quantized


# ============================================================================
# INFERENCE PROFILER
# ============================================================================

class InferenceProfiler:
    """
    Profile model inference performance.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize profiler.
        
        Args:
            model: Model to profile
            device: Device for profiling
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    
    def profile(
        self,
        input_shape: Tuple[int, int, int, int],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Profile inference performance.
        
        Args:
            input_shape: Input tensor shape (B, C, H, W)
            num_iterations: Number of iterations to benchmark
            warmup_iterations: Warmup iterations before benchmarking
            
        Returns:
            Dictionary with timing statistics
        """
        print(f"\nProfiling model on {self.device}...")
        print(f"Input shape: {input_shape}")
        print(f"Iterations: {num_iterations} (+ {warmup_iterations} warmup)")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(dummy_input)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        timings = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc="Benchmarking"):
                start = time.perf_counter()
                _ = self.model(dummy_input)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                timings.append(end - start)
        
        # Calculate statistics
        timings = np.array(timings) * 1000  # Convert to ms
        
        results = {
            'mean_ms': float(np.mean(timings)),
            'std_ms': float(np.std(timings)),
            'min_ms': float(np.min(timings)),
            'max_ms': float(np.max(timings)),
            'median_ms': float(np.median(timings)),
            'p95_ms': float(np.percentile(timings, 95)),
            'p99_ms': float(np.percentile(timings, 99)),
            'throughput_fps': float(1000.0 / np.mean(timings))
        }
        
        # Print results
        print("\n" + "=" * 60)
        print("PROFILING RESULTS")
        print("=" * 60)
        print(f"Mean:       {results['mean_ms']:.2f} ms")
        print(f"Std Dev:    {results['std_ms']:.2f} ms")
        print(f"Min:        {results['min_ms']:.2f} ms")
        print(f"Max:        {results['max_ms']:.2f} ms")
        print(f"Median:     {results['median_ms']:.2f} ms")
        print(f"P95:        {results['p95_ms']:.2f} ms")
        print(f"P99:        {results['p99_ms']:.2f} ms")
        print(f"Throughput: {results['throughput_fps']:.2f} FPS")
        print("=" * 60)
        
        return results
    
    
    def compare_optimizations(
        self,
        optimized_models: Dict[str, nn.Module],
        input_shape: Tuple[int, int, int, int],
        num_iterations: int = 100
    ):
        """
        Compare performance of different optimization strategies.
        
        Args:
            optimized_models: Dictionary of model_name -> model
            input_shape: Input tensor shape
            num_iterations: Number of iterations
        """
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPARISON")
        print("=" * 80)
        
        results = {}
        
        # Profile original model
        print("\n1. Original Model")
        results['original'] = self.profile(input_shape, num_iterations)
        
        # Profile optimized models
        for i, (name, model) in enumerate(optimized_models.items(), 2):
            print(f"\n{i}. {name}")
            profiler = InferenceProfiler(model, self.device)
            results[name] = profiler.profile(input_shape, num_iterations)
        
        # Print comparison
        print("\n" + "=" * 80)
        print("SPEEDUP COMPARISON")
        print("=" * 80)
        
        baseline = results['original']['mean_ms']
        
        for name, metrics in results.items():
            speedup = baseline / metrics['mean_ms']
            print(f"{name:20s}: {metrics['mean_ms']:6.2f} ms  |  {speedup:.2f}x speedup")
        
        print("=" * 80)


# ============================================================================
# MAIN OPTIMIZATION PIPELINE
# ============================================================================

def main():
    """Run full optimization pipeline."""
    
    print("=" * 80)
    print("MODEL OPTIMIZATION PIPELINE")
    print("=" * 80)
    
    # Configuration
    MODEL_PATH = "weights/landmark_detector_best.pth"
    OUTPUT_DIR = Path("optimized_models")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = LandmarkDetector(model_path=MODEL_PATH, device=device)
    model = detector.model
    
    # 1. Export to ONNX
    print("\n" + "=" * 80)
    print("1. ONNX EXPORT")
    print("=" * 80)
    dummy_input = torch.randn(1, 3, 300, 300).to(device)
    export_to_onnx(
        model,
        dummy_input,
        str(OUTPUT_DIR / "model.onnx")
    )
    
    # 2. TensorRT conversion (if available)
    if device == 'cuda':
        print("\n" + "=" * 80)
        print("2. TENSORRT CONVERSION")
        print("=" * 80)
        try:
            convert_to_tensorrt(
                str(OUTPUT_DIR / "model.onnx"),
                str(OUTPUT_DIR / "model_fp16.engine"),
                precision='fp16'
            )
        except Exception as e:
            print(f"TensorRT conversion failed: {e}")
    
    # 3. Pruning
    print("\n" + "=" * 80)
    print("3. MODEL PRUNING")
    print("=" * 80)
    pruned_model = prune_model(model, amount=0.3)
    torch.save(pruned_model.state_dict(), OUTPUT_DIR / "model_pruned.pth")
    
    # 4. Quantization (CPU only)
    if device == 'cpu':
        print("\n" + "=" * 80)
        print("4. MODEL QUANTIZATION")
        print("=" * 80)
        quantized_model = quantize_model(model)
        torch.save(quantized_model.state_dict(), OUTPUT_DIR / "model_quantized.pth")
    
    # 5. Profiling
    print("\n" + "=" * 80)
    print("5. PERFORMANCE PROFILING")
    print("=" * 80)
    
    profiler = InferenceProfiler(model, device)
    optimized_models = {}
    
    if (OUTPUT_DIR / "model_pruned.pth").exists():
        optimized_models['Pruned'] = pruned_model
    
    if device == 'cpu' and (OUTPUT_DIR / "model_quantized.pth").exists():
        optimized_models['Quantized'] = quantized_model
    
    profiler.compare_optimizations(
        optimized_models,
        input_shape=(1, 3, 300, 300),
        num_iterations=50
    )
    
    print("\n✓ Optimization pipeline complete!")
    print(f"Optimized models saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
