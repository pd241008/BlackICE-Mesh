import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from app.ml.models.architecture import TabularMLP
from scripts.variance_audit import pgd_dacm_attack, unconstrained_pgd_attack

def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Loading test dataset...")
    data = np.loadtxt("./data/nsl-kdd-test.csv", delimiter=',')
    X = torch.tensor(data[:, :18], dtype=torch.float32)
    y = torch.tensor(data[:, 18], dtype=torch.long)
    
    loader = DataLoader(TensorDataset(X, y), batch_size=1000, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
    
    model = TabularMLP().to(DEVICE)
    model.load_state_dict(torch.load("models/model_adv.pth", map_location=DEVICE, weights_only=True))
    model.eval()

    from app.ml.data.loader import CATEGORICAL_GROUPS
    bounds = [torch.eye(len(group), dtype=torch.float32) for group in CATEGORICAL_GROUPS]
    
    set_seed(0)
    adv_constrained = pgd_dacm_attack(model, batch_x, batch_y, 0.15, 0.01, 40, bounds, snap_times={"total":0.0, "calls":0})
    
    set_seed(0)
    adv_unconstrained = unconstrained_pgd_attack(model, batch_x, batch_y, 0.15, 0.01, 40)
    
    print("--- Step 2.2: Tensor Equality Check ---")
    print(f"Tensors exactly equal: {torch.equal(adv_constrained, adv_unconstrained)}")
    print(f"Max absolute difference: {(adv_constrained - adv_unconstrained).abs().max().item()}")

    # Step 2.3 Seed-consumption isolation test
    print("\n--- Step 2.3: Seed Isolation Check ---")
    set_seed(0)
    r1 = torch.empty(1000, 4).uniform_(-0.15, 0.15)
    set_seed(0)
    r2 = torch.empty(1000, 4).uniform_(-0.15, 0.15)
    print(f"Random draws exactly equal: {torch.equal(r1, r2)}")

if __name__ == "__main__":
    main()
