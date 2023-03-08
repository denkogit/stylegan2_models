import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim
from arcface_model import get_model



class Arcface_Loss(nn.Module):
    def __init__(self, weights_path, device):
        super().__init__()

        self.arcnet = get_model("r50", fp16=False)
        self.arcnet.load_state_dict(torch.load(weights_path))
        self.arcnet.eval()
        self.arcnet.to(device)
    
        self.cosin_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
      
    def forward(self, target, synth):
        emb1 = self.facenet(target)
        emb2 = self.facenet(synth)
        loss = (1 - self.cosin_loss(emb1, emb2))[0]
        return loss



class Rec_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')


    def calculate(self,generate_tensor, attribute_tensor, mask_tensors):
      
        generate_tensor = torch.add(generate_tensor, 1.0)
        generate_tensor = torch.mul(generate_tensor, 127.5)
        generate_tensor = generate_tensor / 255

        attribute_tensor = torch.add(attribute_tensor, 1.0)
        attribute_tensor = torch.mul(attribute_tensor, 127.5)
        attribute_tensor = attribute_tensor / 255

        loss = torch.mean(1 - ms_ssim(attribute_tensor, generate_tensor, data_range=1, size_average=True))
        return loss
    
    