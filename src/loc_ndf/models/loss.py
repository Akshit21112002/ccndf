import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
#print("loss11")
class ProjectedDistanceLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.1, power=0, plane_dist=True) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.power = power
        self.i = 0
        self.plane_dist = plane_dist

    def forward(self,
                points_distance: torch.Tensor,
                points: torch.Tensor,
                ray_dists: torch.Tensor,
                inter_val: torch.Tensor,
                inter_pos: torch.Tensor,
                inter_grad: torch.Tensor,
                rand_grad: torch.Tensor,
                radius: torch.Tensor):
    
        if self.plane_dist:
            d_pos = inter_pos-points.unsqueeze(1)
            # Dot product
            dist = torch.einsum(
                '...n,...n->...', F.normalize(inter_grad.detach(), dim=-1), d_pos.detach()).abs().sqrt()
            #dist1=dist
            dist = dist.unsqueeze(-1)
        else:
            dist = ray_dists
        #print('inter_val',radius)
        #radius2=F.normalize(inter_grad.detach(), dim=-1)
        radius1=radius[:,:,:,3].unsqueeze(-1)
        radius1=torch.nan_to_num(radius1)
        #print("radius",(radius1))
        rex=torch.sqrt(abs(radius1**2-inter_val**2))
        #print("interval:",(inter_grad))
        #print("dpos:",np.shape(d_pos))
        #print("dist:",np.shape(dist))
        #print("dist1:",np.shape(dist1))
        
        dist1=radius1-rex
        weight = (1e-3 + ray_dists.max() - ray_dists)**self.power
        weight = weight * weight.numel() / weight.sum()
        #inter_loss = F.l1_loss(inter_val, dist, reduction='none')
        #inter_loss = (inter_loss * weight).mean()
        #print('interloss',inter_loss)
        #print(rex)
        inter_loss1 = F.l1_loss(inter_val, dist1, reduction='none')
        inter_loss1=torch.nan_to_num(inter_loss1)
        #print("interloss1",inter_loss1)
        inter_loss1 = (inter_loss1 * weight).mean()
        #torch.nan_to_num(x)
        loss_distance = points_distance.abs().mean()

        loss_gradient = ((rand_grad.norm(dim=-1) - 1).abs()).mean()
        gradient_similarity = (1 - F.cosine_similarity(rand_grad[..., 0, :],
                                                       rand_grad[..., 1, :],
                                                       dim=-1)).mean()

        loss = inter_loss1 + loss_gradient*self.alpha + \
            loss_distance * self.beta + self.gamma * gradient_similarity
        logs = {"distance": loss_distance, "l1_dist": inter_loss1,
                "gradient": loss_gradient, "gradient_sim": gradient_similarity, "loss": loss}
        return loss, logs
