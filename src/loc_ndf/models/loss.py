import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

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
                inter_radius=torch.tensor
                #rand_radius=torch.tensor
                ):
        #print(list(inter_radius.size()))
        if self.plane_dist:
             d_pos = inter_pos-points.unsqueeze(1)
             # Dot product
             dist = torch.einsum(
                 '...n,...n->...', F.normalize(inter_grad.detach(), dim=-1), d_pos.detach()).abs().sqrt()
             dist = dist.unsqueeze(-1)  #locndf
        else:
             dist = ray_dists
        
        d_pos = torch.norm(inter_pos.detach() - points.unsqueeze(1).detach(), p=2,dim=-1, keepdim=True)
        
        if torch.isnan(inter_radius).any():
            print("Warning: radius is None")
        
        inter_product = torch.sum(d_pos.detach() * inter_radius.detach(), dim=-1, keepdim=True)#dot product
        
        if torch.isnan(inter_product).any():
            print("Warning: inter_prod is None")
        
       

        #print(list(d_pos.size()))
        output_tensor = inter_radius.clone()
        # output_tensor = inter_radius
        output_tensor[..., -1] = inter_radius[..., -1]**2

        output_tensor1 = d_pos.clone()
        # output_tensor1 = d_pos
        output_tensor1[..., -1] = d_pos[..., -1]**2

        output_tensor3 = inter_product.clone()
        # output_tensor3 = inter_product
        output_tensor3[..., -1] = inter_product[..., -1]*2
        rex=torch.sqrt(abs(output_tensor.detach()+output_tensor1.detach()-(2*output_tensor3.detach()))).detach()
        if torch.isnan(rex).any():
            print("Warning: rex is None")
        #print(rex)
        dist1=inter_radius.detach()-rex.detach()# circle distance

        if torch.isnan(dist1).any():
            print("Warning: dist1 is None")
        

        
        dist2=d_pos #ray_dists
        
        if torch.isnan(inter_val).any():
            print("Warning: val is None")
        weight = (1e-3 + ray_dists.max() - ray_dists)**self.power
        weight = weight * weight.numel() / weight.sum()
        
        inter_loss = F.l1_loss((inter_val), dist1, reduction='none')

       

        inter_loss = ((inter_loss * weight).mean()) 
        # print(inter_loss)


        loss_distance = points_distance.abs().mean()

        loss_gradient = ((rand_grad.norm(dim=-1) - 1).abs()).mean()
        gradient_similarity = (1 - F.cosine_similarity(rand_grad[..., 0, :],
                                                       rand_grad[..., 1, :],
                                                       dim=-1)).mean()

        
        loss = inter_loss + loss_gradient*self.alpha + \
            loss_distance * self.beta  + self.gamma * gradient_similarity 
        

        logs = {"distance": loss_distance, "l1_dist": inter_loss,
                 "gradient": loss_gradient, "gradient_sim": gradient_similarity, "loss": loss}

        
        return loss, logs
