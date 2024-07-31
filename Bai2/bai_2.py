import numpy as np

# a
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def triplet_loss(anchor, positive, negative, alpha=0.1):
    # Tính khoảng cách Euclidean
    d_ap = euclidean_distance(anchor, positive)
    d_an = euclidean_distance(anchor, negative)
    
    # Tính Triplet loss
    loss = np.maximum(0, d_ap - d_an + alpha)
    
    return loss



# b
def multi_triplet_loss(anchor, positives, negatives, alpha=0.1):
    # Tính Triplet loss tổng hợp
    loss = 0
    for p in positives:
        for n in negatives:
            d_ap = euclidean_distance(anchor, p)
            d_an = euclidean_distance(anchor, n)
            loss += np.maximum(0, d_ap - d_an + alpha)
    
    return loss
