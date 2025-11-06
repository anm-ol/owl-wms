import torch

# Path to your file
path = "/home/sky/summer193/owl-wms/rgb_latents/round_007/000000_rgblatent.pt"

# Load the file
print(f"Loading {path}...")
data = torch.load(path, map_location="cpu")

print("\nFile loaded successfully")

if isinstance(data, dict):
    print(f"\nType: dict with {len(data)} keys\n")
    for key, value in data.items():
        print(f"Key: {key}")
        if torch.is_tensor(value):
            print(f"  • Tensor shape: {tuple(value.shape)}")
            print(f"  • Dtype: {value.dtype}")
            print(f"  • Min/Max: {value.min().item():.4f} / {value.max().item():.4f}")
        else:
            print(f"  • Type: {type(value)}")
        print("-" * 40)

elif torch.is_tensor(data):
    print(f"\nType: single tensor")
    print(f"Shape: {tuple(data.shape)}")
    print(f"Dtype: {data.dtype}")
    print(f"Min/Max: {data.min().item():.4f} / {data.max().item():.4f}")

else:
    print(f"\nType: {type(data)}")
    print("Contents preview:")
    print(data)
