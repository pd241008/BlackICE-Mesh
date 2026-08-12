import torch
import torch.nn as nn
import torch.nn.functional as F

def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, steps=40, continuous_cols=None, categorical_groups=None):
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    
    if continuous_cols is None:
        continuous_cols = list(range(images.shape[1]))
    if categorical_groups is None:
        categorical_groups = []
        
    ori_images = images.clone().detach()
    
    for i in range(steps):
        images.requires_grad = True
        outputs = model(images)
        
        model.zero_grad()
        cost = loss_fn(outputs, labels)
        cost.backward()
        
        grad = images.grad
        
        # 1. Continuous Perturbation & Projection (L_inf and Min-Max [0, 1])
        if continuous_cols:
            adv_cont = images[:, continuous_cols] + alpha * grad[:, continuous_cols].sign()
            
            # L_inf Projection: bounded to epsilon ball
            eta = torch.clamp(adv_cont - ori_images[:, continuous_cols], min=-epsilon, max=epsilon)
            
            # Min-Max Constraint: bounded to [0, 1]
            adv_cont_snapped = torch.clamp(ori_images[:, continuous_cols] + eta, min=0.0, max=1.0)
            
            images.data[:, continuous_cols] = adv_cont_snapped
            
        # 2. Categorical Constraints (DACM) using L_2 Euclidean Nearest Neighbor (argmax)
        for cat_group in categorical_groups:
            # Apply continuous gradient step
            adv_cat = images[:, cat_group] + alpha * grad[:, cat_group].sign()
            
            # Find Euclidean nearest neighbor in one-hot space (argmax)
            nearest_idx = torch.argmax(adv_cat, dim=1)
            
            # Snap tensor back to valid discrete structure
            snapped_tensor = F.one_hot(nearest_idx, num_classes=len(cat_group)).float()
            
            # Inject structurally valid payload back into main tensor
            images.data[:, cat_group] = snapped_tensor
            
        images = images.detach()
        
    return images
