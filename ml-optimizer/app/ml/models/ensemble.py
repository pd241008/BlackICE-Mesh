import torch
import torch.nn as nn
from app.ml.models.architecture import TabularMLP


class EnsembleModel(nn.Module):
    def __init__(self, num_models=3):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models = nn.ModuleList()
        self.num_models = num_models

        for i in range(num_models):
            path = f"models/model_{i}.pth"

            try:
                model = TabularMLP().to(self.device)
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                self.models.append(model)
                print(f"Loaded {path}")

            except Exception as e:
                print(f"Failed to load {path}: {e}")

        if len(self.models) == 0:
            raise ValueError("No models loaded in ensemble")

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        stacked = torch.stack(outputs)
        avg_output = torch.mean(stacked, dim=0)
        return avg_output
