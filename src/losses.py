import torch
import torch.nn.functional as F

def clip_loss(logits_per_image):
    """Standard symmetric InfoNCE, the CLIP baseline."""
    bs = logits_per_image.size(0)
    labels = torch.arange(bs, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_image.T, labels)
    return (loss_i + loss_t) / 2


def masked_infonce(logits_per_image, sim_matrix, threshold=0.5):
    """
    mask false negatives out of the softmax denominator.
    sim_matrix: (bs, bs) float tensor, Jaccard similarities for this batch.
    """
    bs = logits_per_image.size(0)
    labels = torch.arange(bs, device=logits_per_image.device)

    #build mask: off-diagonal entries where similarity > threshold
    mask = (sim_matrix > threshold).float()
    mask.fill_diagonal_(0)   # never mask the true positive

    #push masked positions to -inf so they vanish from softmax
    neg_inf = torch.full_like(logits_per_image, float("-inf"))
    logits_masked = torch.where(mask.bool(), neg_inf, logits_per_image)

    loss_i = F.cross_entropy(logits_masked,   labels)
    loss_t = F.cross_entropy(logits_masked.T, labels)
    return (loss_i + loss_t) / 2


def soft_target_infonce(logits_per_image, sim_matrix, tau=0.5):
    """
    soft label targets (MedCLIP-style).
    Instead of a one-hot target, use the similarity distribution as the target.
    tau controls how peaked the target distribution is.
    """
    #build soft targets: force diagonal to 1 first, then softmax
    targets = sim_matrix.clone()
    targets.fill_diagonal_(1.0)
    targets = F.softmax(targets / tau, dim=-1)

    log_probs_i = F.log_softmax(logits_per_image,   dim=-1)
    log_probs_t = F.log_softmax(logits_per_image.T, dim=-1)

    loss_i = -(targets * log_probs_i).sum(dim=-1).mean()
    loss_t = -(targets * log_probs_t).sum(dim=-1).mean()
    return (loss_i + loss_t) / 2