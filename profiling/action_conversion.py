import torch
import time

def action_id_to_buttons_original(action_id: torch.Tensor):
    """Original version - potentially slow"""
    # Create a tensor for bit positions [0, 1, 2, 3, 4, 5, 6, 7]
    bit_positions = torch.arange(8, device=action_id.device, dtype=action_id.dtype)
    
    # Expand dimensions for broadcasting: [B, N] -> [B, N, 1] and [8] -> [1, 1, 8]
    action_expanded = action_id.unsqueeze(-1)  # [B, N, 1]
    bit_positions = bit_positions.unsqueeze(0).unsqueeze(0)  # [1, 1, 8]
    
    # Right shift action_id by each bit position and check the least significant bit
    buttons = (action_expanded >> bit_positions) & 1
    
    return buttons.int()

def action_id_to_buttons_optimized_v1(action_id: torch.Tensor):
    """Optimized version 1 - Pre-compute bit_positions"""
    # Pre-compute and cache bit positions on the same device
    if not hasattr(action_id_to_buttons_optimized_v1, '_bit_positions_cache'):
        action_id_to_buttons_optimized_v1._bit_positions_cache = {}
    
    device_key = (action_id.device, action_id.dtype)
    if device_key not in action_id_to_buttons_optimized_v1._bit_positions_cache:
        bit_positions = torch.arange(8, device=action_id.device, dtype=action_id.dtype)
        bit_positions = bit_positions.view(1, 1, 8)  # Pre-shape for broadcasting
        action_id_to_buttons_optimized_v1._bit_positions_cache[device_key] = bit_positions
    
    bit_positions = action_id_to_buttons_optimized_v1._bit_positions_cache[device_key]
    
    # Expand action_id for broadcasting
    action_expanded = action_id.unsqueeze(-1)
    
    # Bit operations
    buttons = (action_expanded >> bit_positions) & 1
    
    return buttons.int()

def action_id_to_buttons_optimized_v2(action_id: torch.Tensor):
    """Optimized version 2 - Use powers of 2 instead of bit shifts"""
    # Pre-compute powers of 2: [1, 2, 4, 8, 16, 32, 64, 128]
    if not hasattr(action_id_to_buttons_optimized_v2, '_powers_cache'):
        action_id_to_buttons_optimized_v2._powers_cache = {}
    
    device_key = (action_id.device, action_id.dtype)
    if device_key not in action_id_to_buttons_optimized_v2._powers_cache:
        powers = torch.tensor([2**i for i in range(8)], 
                             device=action_id.device, dtype=action_id.dtype)
        powers = powers.view(1, 1, 8)
        action_id_to_buttons_optimized_v2._powers_cache[device_key] = powers
    
    powers = action_id_to_buttons_optimized_v2._powers_cache[device_key]
    
    # Use floor division and modulo instead of bit operations
    action_expanded = action_id.unsqueeze(-1)
    buttons = (action_expanded // powers) % 2
    
    return buttons.int()

def action_id_to_buttons_optimized_v3(action_id: torch.Tensor):
    """Optimized version 3 - Most efficient bit manipulation"""
    # Create bit positions tensor once
    bit_positions = torch.arange(8, device=action_id.device, dtype=torch.long)
    
    # Reshape for broadcasting: [B, N, 1] vs [8]
    action_expanded = action_id.unsqueeze(-1)  # [B, N, 1]
    
    # Use bitwise operations efficiently
    buttons = (action_expanded >> bit_positions) & 1
    
    return buttons.to(torch.uint8)  # Use uint8 instead of int32

def action_id_to_buttons_lookup_table(action_id: torch.Tensor):
    """Version 4 - Pre-computed lookup table (fastest for repeated calls)"""
    if not hasattr(action_id_to_buttons_lookup_table, '_lookup_table'):
        # Pre-compute all possible 8-bit combinations (256 entries)
        lookup = torch.zeros(256, 8, dtype=torch.uint8)
        for i in range(256):
            for j in range(8):
                lookup[i, j] = (i >> j) & 1
        action_id_to_buttons_lookup_table._lookup_table = lookup
    
    lookup_table = action_id_to_buttons_lookup_table._lookup_table.to(action_id.device)
    
    # Use advanced indexing - flatten, lookup, reshape
    original_shape = action_id.shape
    flat_actions = action_id.flatten()
    
    # Clamp to valid range [0, 255] to avoid index errors
    flat_actions = torch.clamp(flat_actions, 0, 255).long()
    
    buttons_flat = lookup_table[flat_actions]  # [total_elements, 8]
    buttons = buttons_flat.view(*original_shape, 8)  # [B, N, 8]
    
    return buttons

def benchmark_action_conversions():
    """Benchmark all versions"""
    print("=== Benchmarking Action Conversion Functions ===")
    
    # Test with realistic data shapes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different input sizes
    test_sizes = [
        (1, 1, 8),      # Single frame
        (1, 16, 8),     # Window
        (4, 16, 8),     # Batch
        (1, 60, 8),     # Full sequence
    ]
    
    functions = [
        ("Original", action_id_to_buttons_original),
        ("Optimized V1 (cached)", action_id_to_buttons_optimized_v1),
        ("Optimized V2 (powers)", action_id_to_buttons_optimized_v2),
        ("Optimized V3 (efficient)", action_id_to_buttons_optimized_v3),
        ("Lookup Table", action_id_to_buttons_lookup_table),
    ]
    
    for size in test_sizes:
        print(f"\n--- Testing shape {size} ---")
        
        # Create random action IDs (0-255 range for valid 8-bit values)
        action_ids = torch.randint(0, 256, size, device=device)
        
        # Warm up GPU
        for _ in range(10):
            for _, func in functions:
                _ = func(action_ids)
        
        # Benchmark each function
        times = {}
        for name, func in functions:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            
            start_time = time.perf_counter()
            for _ in range(100):  # 100 iterations
                result = func(action_ids)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) * 1000 / 100  # ms per call
            times[name] = avg_time
            
            print(f"  {name:20s}: {avg_time:.4f}ms per call")
        
        # Show speedup vs original
        original_time = times["Original"]
        fastest_name = min(times, key=times.get)
        fastest_time = times[fastest_name]
        
        print(f"  Fastest: {fastest_name} ({original_time/fastest_time:.1f}x speedup)")

def estimate_sampling_impact():
    """Estimate impact on sampling performance"""
    print("\n=== Estimated Impact on Sampling ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    single_action = torch.randint(0, 256, (1, 1, 8), device=device)
    
    # Time original vs fastest method
    num_calls = 1000
    
    # Original
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = action_id_to_buttons_original(single_action)
    original_total = (time.perf_counter() - start) * 1000
    
    # Lookup table (likely fastest)
    start = time.perf_counter()
    for _ in range(num_calls):
        _ = action_id_to_buttons_lookup_table(single_action)
    lookup_total = (time.perf_counter() - start) * 1000
    
    original_per_call = original_total / num_calls
    lookup_per_call = lookup_total / num_calls
    
    print(f"Original method: {original_per_call:.4f}ms per call")
    print(f"Lookup table: {lookup_per_call:.4f}ms per call")
    print(f"Speedup: {original_per_call/lookup_per_call:.1f}x")
    
    # Calculate impact on 60-frame sampling (2 calls per frame)
    frames = 60
    calls_per_frame = 2
    
    original_overhead = original_per_call * frames * calls_per_frame
    lookup_overhead = lookup_per_call * frames * calls_per_frame
    savings = original_overhead - lookup_overhead
    
    print(f"\nFor 60-frame sampling (120 total calls):")
    print(f"  Original overhead: {original_overhead:.1f}ms")
    print(f"  Optimized overhead: {lookup_overhead:.1f}ms")
    print(f"  Time savings: {savings:.1f}ms")
    
    if savings > 100:
        print(f"  ✅ Significant savings! This could explain part of the slowdown.")
    elif savings > 20:
        print(f"  ⚠️  Moderate savings. Helpful but may not be the main bottleneck.")
    else:
        print(f"  ℹ️  Minor savings. Look for other bottlenecks.")

if __name__ == "__main__":
    benchmark_action_conversions()
    estimate_sampling_impact()