import torch
import numpy as np


def pairwise_distances(a, b=None, eps=1e-6):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    :param a:
    :param b:
    :return:
    """
    if b is None:
        b = a

    aa = torch.sum(a**2, dim=1)
    bb = torch.sum(b**2, dim=1)

    aa = aa.expand(bb.size(0), aa.size(0)).t()
    bb = bb.expand(aa.size(0), bb.size(0))

    AB = torch.mm(a, b.transpose(0, 1))

    dists = aa + bb - 2 * AB
    dists = torch.clamp(dists, min=0, max=np.inf)
    dists = torch.sqrt(dists + eps)
    return dists


def flac_loss(protected_attr_features, features, labels, d=1):
    # Protected attribute features kernel
    protected_d = pairwise_distances(protected_attr_features)
    protected_s = 1.0 / (1 + protected_d**d)
    # Target features kernel
    features_d = pairwise_distances(features)
    features_s = 1.0 / (1 + features_d**d)

    th = (torch.max(protected_s) + torch.min(protected_s)) / 2
    # calc the mask
    mask = (labels[:, None] == labels) & (protected_s < th) | (
        labels[:, None] != labels
    ) & (protected_s > th)
    mask = mask.to(labels.device)

    # if mask is empty, return zero
    if sum(sum(mask)) == 0:
        return torch.tensor(0.0).to(labels.device)
    # similarity to distance
    protected_s = 1 - protected_s

    # convert to probabilities
    protected_s = protected_s / (
        torch.sum(protected_s * mask.int().float(), dim=1, keepdim=True) + 1e-7
    )
    features_s = features_s / (
        torch.sum(features_s * mask.int().float(), dim=1, keepdim=True) + 1e-7
    )

    # Jeffrey's divergence
    loss = (protected_s[mask] - features_s[mask]) * (
        torch.log(protected_s[mask]) - torch.log(features_s[mask])
    )

    return torch.mean(loss)

def flac_loss_multilabel(protected_attr_features, features, labels, d=1, similarity_threshold=0.5):
    """
    Compute the FLAC loss adapted for multilabel classification.
    
    Args:
        protected_attr_features (torch.Tensor): Features related to protected attributes (e.g., race, gender).
        features (torch.Tensor): Main feature set.
        labels (torch.Tensor): Binary matrix of multilabels (shape: [n_samples, n_classes]).
        d (int, optional): Parameter controlling the kernel's decay. Default is 1.
        similarity_threshold (float, optional): Threshold for label similarity (default is 0.5).
        
    Returns:
        torch.Tensor: The computed loss.
    """
    # Protected attribute features kernel
    protected_d = pairwise_distances(protected_attr_features)
    protected_s = 1.0 / (1 + protected_d**d)
    
    # Target features kernel
    features_d = pairwise_distances(features)
    features_s = 1.0 / (1 + features_d**d)
    
    # Calculate label similarity using Jaccard index
    #intersection = (labels[:, None, :] & labels[None, :, :]).sum(dim=2).float()
    intersection = ((labels > 0.5).bool()[:, None, :] & (labels > 0.5).bool()[None, :, :]).sum(dim=2).float()
    labels_bool = (labels > 0.5).bool()  # Thresholding for float labels

    # Compute union using boolean labels
    union = (labels_bool[:, None, :] | labels_bool[None, :, :]).sum(dim=2).float()
    #union = (labels[:, None, :] | labels[None, :, :]).sum(dim=2).float()
    label_similarity = intersection / (union + 1e-7)
    
    # Create the mask for multilabel case
    same_label_mask = label_similarity >= similarity_threshold
    different_label_mask = label_similarity < similarity_threshold
    th = (torch.max(protected_s) + torch.min(protected_s)) / 2
    mask = (same_label_mask & (protected_s < th)) | (different_label_mask & (protected_s > th))
    mask = mask.to(labels.device)
    
    # If mask is empty, return zero
    if torch.sum(mask) == 0:
        return torch.tensor(0.0, device=labels.device)
    
    # Convert similarity to distance
    protected_s = 1 - protected_s
    
    # Convert to probabilities
    protected_s = protected_s / (
        torch.sum(protected_s * mask.float(), dim=1, keepdim=True) + 1e-7
    )
    features_s = features_s / (
        torch.sum(features_s * mask.float(), dim=1, keepdim=True) + 1e-7
    )
    
    # Jeffrey's divergence
    loss = (protected_s[mask] - features_s[mask]) * (
        torch.log(protected_s[mask] + 1e-7) - torch.log(features_s[mask] + 1e-7)
    )
    
    return torch.mean(loss)
