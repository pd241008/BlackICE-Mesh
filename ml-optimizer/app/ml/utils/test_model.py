import torch
from app.ml.evaluation.base_eval import evaluate
from app.ml.data.loader import get_test_loader
from app.ml.models.architecture import TabularMLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = TabularMLP().to(DEVICE)
model.load_state_dict(torch.load("app/ml/model.pth", map_location=DEVICE))
model.eval()

# Load test data
test_loader = get_test_loader()

# Evaluate
accuracy = evaluate(model, test_loader)

print(f"✅ Clean Accuracy: {accuracy:.4f}")