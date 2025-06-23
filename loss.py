import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def iq2d(iq):
    theta = torch.atan2(iq[:,0:1,:,:], iq[:,1:2,:,:])
    theta[torch.lt(theta, 0)] = theta[torch.lt(theta, 0)]+2*np.pi
    depth = (theta / (2*np.pi) )
   
    return depth
    

class GLoss(nn.Module):
    def __init__(self, device, weight=0.05):
        super(GLoss, self).__init__()
        self.weight = weight
        self.max_range_iq = 0.15
        self.max_range_d = 1
        self.device = device

    def forward(self, out_0, out_1, out_2, ideal_IQ, ideal_d):
        """
        :param out: [batch_size, 2, H, W]*3
        :param ideal: [batch_size, 2, H, W]
        :return:
        """
        ideal_IQ_0 = ideal_IQ[:,0:2,:,:]
        ideal_IQ_1 = ideal_IQ[:,2:4,:,:]
        ideal_IQ_2 = ideal_IQ[:,4:6,:,:]

        d_mask = (ideal_d != 0) * (ideal_d < 10) 
        iq_mask = torch.concat([d_mask,d_mask], axis=1)
        # and (coarse_depth != 0) and (fine_depth != 0) and (coarse_depth < 10) and (fine_depth < 10)
        other_d = torch.ones(ideal_d.shape).contiguous().to(self.device) 
        other_iq = torch.ones(ideal_IQ_0.shape).contiguous().to(self.device)

        iq_loss_0 = torch.min(torch.abs(out_0[iq_mask] - ideal_IQ_0[iq_mask]), self.max_range_iq*other_iq[iq_mask]).mean()
        iq_loss_1 = torch.min(torch.abs(out_1[iq_mask] - ideal_IQ_1[iq_mask]), self.max_range_iq*other_iq[iq_mask]).mean()
        iq_loss_2 = torch.min(torch.abs(out_2[iq_mask] - ideal_IQ_2[iq_mask]), self.max_range_iq*other_iq[iq_mask]).mean()
        
        d_0 = iq2d(out_0)
        d_1 = iq2d(out_1)
        d_2 = iq2d(out_2)
        d_ideal_0 = iq2d(ideal_IQ_0)
        d_ideal_1 = iq2d(ideal_IQ_1)
        d_ideal_2 = iq2d(ideal_IQ_2)
        d_loss_0 = torch.min(torch.abs(d_0[d_mask] - d_ideal_0[d_mask]), self.max_range_d*other_d[d_mask]).mean()
        d_loss_1 = torch.min(torch.abs(d_1[d_mask] - d_ideal_1[d_mask]), self.max_range_d*other_d[d_mask]).mean()
        d_loss_2 = torch.min(torch.abs(d_2[d_mask] - d_ideal_2[d_mask]), self.max_range_d*other_d[d_mask]).mean()
        
        """ L1 loss """
        loss_sup = iq_loss_0 + iq_loss_1 + iq_loss_2 
        loss_sup_d = d_loss_0 + d_loss_1 + d_loss_2
        
        return loss_sup + 0.1 * loss_sup_d


class GLoss_test(nn.Module):
    def __init__(self, weight=0.05):
        super(GLoss_test, self).__init__()
        self.weight = weight
        self.max_range_iq = 0.15
        self.max_range_d = 1

    def forward(self, out_0, out_1, out_2, ideal_IQ, ideal_d):
        """
        :param out: [batch_size, 2, H, W]*3
        :param ideal: [batch_size, 2, H, W]
        :return:
        """
        ideal_IQ_0 = ideal_IQ[:,0:2,:,:]
        ideal_IQ_1 = ideal_IQ[:,2:4,:,:]
        ideal_IQ_2 = ideal_IQ[:,4:6,:,:]

        d_mask = (ideal_d != 0) * (ideal_d < 10) 
        iq_mask = torch.concat([d_mask,d_mask], axis=1)
        # and (coarse_depth != 0) and (fine_depth != 0) and (coarse_depth < 10) and (fine_depth < 10)

        iq_loss_0 = torch.abs(out_0[iq_mask] - ideal_IQ_0[iq_mask]).mean()
        iq_loss_1 = torch.abs(out_1[iq_mask] - ideal_IQ_1[iq_mask]).mean()
        iq_loss_2 = torch.abs(out_2[iq_mask] - ideal_IQ_2[iq_mask]).mean()
        
        d_0 = iq2d(out_0)
        d_1 = iq2d(out_1)
        d_2 = iq2d(out_2)
        d_ideal_0 = iq2d(ideal_IQ_0)
        d_ideal_1 = iq2d(ideal_IQ_1)
        d_ideal_2 = iq2d(ideal_IQ_2)
        d_loss_0 = torch.abs(d_0[d_mask] - d_ideal_0[d_mask]).mean()
        d_loss_1 = torch.abs(d_1[d_mask] - d_ideal_1[d_mask]).mean()
        d_loss_2 = torch.abs(d_2[d_mask] - d_ideal_2[d_mask]).mean()
        

        """ L1 loss """
        loss_sup = iq_loss_0 + iq_loss_1 + iq_loss_2 
        loss_sup_d = d_loss_0 + d_loss_1 + d_loss_2
        
        return loss_sup + 0.1 * loss_sup_d