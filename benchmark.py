import torch
import time
from model_dfs import DynamicFocusNet

def benchmark_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        
    model = DynamicFocusNet(num_classes=3).to(device)
    model.eval()
    
    # Dummy input (B, C, H, W)
    input_tensor = torch.randn(1, 3, 512, 512).to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
            
    # Measure
    num_runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs * 1000 # ms
    
    print(f"Average inference time for 512x512 image on {device}: {avg_time:.2f} ms")
    
    if avg_time <= 200:
        print("Success: Inference time <= 200ms")
    else:
        print("Warning: Inference time > 200ms (Note: Performance depends on hardware)")

if __name__ == '__main__':
    benchmark_inference()
