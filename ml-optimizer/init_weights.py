import torch
from app.ml.models.architecture import TabularMLP

def init_dummy_weights():
    model = TabularMLP()
    paths = ["app/ml/model.pth", "app/ml/model_0.pth", "app/ml/model_1.pth", "app/ml/model_2.pth"]
    for p in paths:
        torch.save(model.state_dict(), p)
        print(f"Saved dummy weights to {p}")

if __name__ == "__main__":
    init_dummy_weights()
