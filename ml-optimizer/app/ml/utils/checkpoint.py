import torch
import hashlib
import os

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_model_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    hash_val = compute_sha256(path)
    print(f"[CHECKPOINT] Saved model to {path}")
    print(f"[CHECKPOINT] SHA-256: {hash_val}")
    return hash_val

def load_model_checkpoint(model, path, device=None):
    if device is None:
        device = torch.device("cpu")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    hash_val = compute_sha256(path)
    print(f"[CHECKPOINT] Loaded model from {path}")
    print(f"[CHECKPOINT] SHA-256: {hash_val}")
    return hash_val
