import torch
import torch.nn as nn
import torch.nn.functional as F

def unified_pgd_attack(model, images, labels, epsilon=0.15, alpha=0.01, alpha_cat=None, steps=10, continuous_cols=None, categorical_groups=None, rsc=False):
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = nn.CrossEntropyLoss()
    
    current_alpha_cat = alpha if alpha_cat is None else alpha_cat
    
    if continuous_cols is None:
        continuous_cols = []
    if categorical_groups is None:
        categorical_groups = []
        
    ori_images = images.clone().detach()
    
    num_groups = len(categorical_groups)
    rsc_mask = None
    if rsc and num_groups > 0:
        k_samples = torch.randint(0, num_groups + 1, (images.size(0),), device=images.device)
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
        
        if continuous_cols:
            adv_cont = images[:, continuous_cols] + alpha * grad[:, continuous_cols].sign()
            eta = torch.clamp(adv_cont - ori_images[:, continuous_cols], min=-epsilon, max=epsilon)
            adv_cont_snapped = torch.clamp(ori_images[:, continuous_cols] + eta, min=0.0, max=1.0)
            images.data[:, continuous_cols] = adv_cont_snapped
            
        if current_alpha_cat > 0:
            for g_idx, cat_group in enumerate(categorical_groups):
                adv_cat = images[:, cat_group] + current_alpha_cat * grad[:, cat_group].sign()
                nearest_idx = torch.argmax(adv_cat, dim=1)
                snapped_tensor = F.one_hot(nearest_idx, num_classes=len(cat_group)).float()
                
                if rsc and rsc_mask is not None:
                    active_mask = rsc_mask[:, g_idx].unsqueeze(1)
                    images.data[:, cat_group] = torch.where(active_mask, snapped_tensor, ori_images[:, cat_group])
                else:
                    images.data[:, cat_group] = snapped_tensor
            
        images = images.detach()
        
    return images
