import torch
import torch.nn as nn
import torch.nn.functional as F
def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, steps=40, continuous_cols=None, categorical_groups=None, rsc=False):
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    
    if continuous_cols is None:
        continuous_cols = []
    if categorical_groups is None:
        categorical_groups = []
        
    ori_images = images.clone().detach()
    
    num_groups = len(categorical_groups)
    rsc_mask = None
    if rsc and num_groups > 0:
        # Uniform sampling per-sample k in [0, num_groups]
        # E.g. for CICIDS2017 (num_groups=1), k is sampled from {0, 1}
        # For UNSW-NB15 (num_groups=2), k is sampled from {0, 1, 2}
        # If num_groups grows significantly, consider revising this to ensure k=1 remains adequately represented.
        k_samples = torch.randint(0, num_groups + 1, (images.size(0),), device=images.device)
        
        if not getattr(pgd_attack, "rsc_logged", False):
            unique, counts = torch.unique(k_samples, return_counts=True)
            dist = {int(k): int(c) for k, c in zip(unique, counts)}
            print(f"[RSC] Sampled k distribution (first batch): {dist}")
            pgd_attack.rsc_logged = True
            
        rand_vals = torch.rand(images.size(0), num_groups, device=images.device)
        ranks = rand_vals.argsort(dim=1).argsort(dim=1)
        rsc_mask = ranks < k_samples.unsqueeze(1)
    
    if continuous_cols:
        random_noise = torch.empty_like(ori_images[:, continuous_cols]).uniform_(-epsilon, epsilon)
        images[:, continuous_cols] = torch.clamp(ori_images[:, continuous_cols] + random_noise, 0.0, 1.0)
        
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
        for g_idx, cat_group in enumerate(categorical_groups):
            # Apply continuous gradient step
            adv_cat = images[:, cat_group] + alpha * grad[:, cat_group].sign()
            
            # Find Euclidean nearest neighbor in one-hot space (argmax)
            nearest_idx = torch.argmax(adv_cat, dim=1)
            
            # Snap tensor back to valid discrete structure
            snapped_tensor = F.one_hot(nearest_idx, num_classes=len(cat_group)).float()
            
            if rsc and rsc_mask is not None:
                active_mask = rsc_mask[:, g_idx].unsqueeze(1)
                # Strict freezing: explicitly re-apply ori_images to unselected groups every step
                images.data[:, cat_group] = torch.where(active_mask, snapped_tensor, ori_images[:, cat_group])
            else:
                # Inject structurally valid payload back into main tensor
                images.data[:, cat_group] = snapped_tensor
            
        images = images.detach()
        
    return images
