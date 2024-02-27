import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
import scienceplots
def similarity_cost(x1, x2, gamma=10):
    '''
    :param x1: [B N D]
    :param x2: [B M D]
    :return: [B N M]
    '''
    N = x1.shape[1]
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    sim = torch.bmm(x1, x2.transpose(1, 2)) * gamma
    sim = sim.exp()
    sim_c = sim.sum(dim=1, keepdim=True)
    sim_r = sim.sum(dim=2, keepdim=True)
    sim = sim / (sim_c + sim_r - sim)
    return 1 - sim

def get_group_ones(N, b, B, device):
    alpha = []
    for i in range(B):
        if i == b:
            alpha.append(torch.ones(1, N[i], device=device).float())
        else:
            alpha.append(torch.zeros(1, N[i], device=device).float())

    alpha = torch.cat(alpha, dim=1)
    return alpha

class GML(nn.Module):
    def __init__(self):
        super().__init__()
        self.ot = SamplesLoss(backend="tensorized", cost=similarity_cost, debias=False,diameter=3)
        self.margin=0.1
    def forward(self, x_pre, x_cur):
        '''
        :param x1: B tensor of size [N_b D]
        :param x2: B tensor of size [N_b D]
        :param x3: B tensor of size [N1_b D] negative exapmles
        :param x4: B tensor of size [M1_b D] negative exapmles
        :return: ot loss
        '''
        x1, x3 = x_pre
        x2, x4 = x_cur
        B = len(x1)  # assert B == 1
        N = [x.shape[0] for x in x1]
        M = [x.shape[0] for x in x2]
        N1 = [x.shape[0] for x in x3]
        M1 = [x.shape[0] for x in x4]

        device = x1[0].device
        ot_loss = []

        for b in range(B):
            assert N[b] == M[b]
            alpha = torch.cat((torch.ones(1, N[b], device=device), torch.zeros(1, N1[b], device=device)), dim=1)
            beta = torch.cat((torch.ones(1, N[b], device=device), torch.zeros(1, M1[b], device=device)), dim=1)
            f1 = torch.cat((x1[b], x3[b]), dim=0).unsqueeze(0)
            f2 = torch.cat((x2[b], x4[b]), dim=0).unsqueeze(0)
            loss = self.ot(alpha, f1, beta, f2)
            ot_loss.append(loss)
        loss=0
        loss+=torch.relu(self.margin-torch.bmm(x3,x4.transpose(1,2))).sum()
        loss_dict={
            "hinge_cost":loss.unsqueeze(0),
            "scon_cost":sum(ot_loss) / len(ot_loss)
        }
        return loss_dict


if __name__ == "__main__":
    # x1 = torch.ones(2, 2)
    # x2 = torch.ones(2, 2)
    from matplotlib import pyplot as plt
    Dim = 512

    x1 = torch.randn(1,25, Dim)
    x2 = torch.randn(1,25, Dim)
    x1.requires_grad = True
    x2.requires_grad = True

    x3 = torch.randn(1,6, Dim)
    x4 = torch.randn(1,9, Dim)
    x3.requires_grad = True
    x4.requires_grad = True

    # x1 = [torch.eye(10)]
    # x2 = [torch.eye(10)]

    lr = 0.5
    loss = GML()
    for i in range(50):
        x1=F.normalize(x1,dim=-1)
        x2=F.normalize(x2,dim=-1)
        x3=F.normalize(x3,dim=-1)
        x4=F.normalize(x4,dim=-1)
        
        l = loss([x1, x3], [x2, x4])
        t=l["scon_cost"]+0.2*l["hinge_cost"]
        if (i+1) % 1 == 0 or i == 0:
            
            s1=torch.bmm(x1,x2.transpose(1,2))
            s2= torch.bmm(x1,x4.transpose(1,2))
            s3 = torch.bmm(x3,x2.transpose(1,2))
            s4 = torch.bmm(x3,x4.transpose(1,2))
            s1_s2=torch.cat((s1,s2),dim=2)
            s3_s4=torch.cat((s3,s4),dim=2)
            img=torch.cat((s1_s2,s3_s4),dim=1)

        [g1, g2, g3, g4] = torch.autograd.grad(t, [x1, x2, x3, x4])
        x1.data -= lr * g1
        x1.data = F.relu(x1.data)
        x2.data -= lr * g2
        x2.data = F.relu(x2.data)
        x3.data -= lr * g3
        x3.data = F.relu(x3.data)
        x4.data -= lr * g4
        x4.data = F.relu(x4.data)
        
    pos_sim_cost=1- similarity_cost(x1, x2)
    neg_sim_cost=1- similarity_cost(x1, x4)
    all_cost=1- similarity_cost(torch.cat((x1,x3),dim=1), torch.cat((x2,x4),dim=1))
    print("pos_sim_cost")
    print(pos_sim_cost.cpu().detach().numpy()[0])
    print("neg_sim_cost")
    print(neg_sim_cost.cpu().detach().numpy()[0])
    print("all_cost")
    print(all_cost.cpu().detach().numpy()[0])
    x1=F.normalize(x1,dim=-1)
    x2=F.normalize(x2,dim=-1)
    x3=F.normalize(x3,dim=-1)
    x4=F.normalize(x4,dim=-1)
    match_matrix=torch.bmm(torch.cat((x1,x3),dim=1),torch.cat((x2,x4),dim=1).transpose(1,2))

    print("match_matrix")
    print(match_matrix.cpu().detach().numpy()[0])
    match=linear_sum_assignment(1-match_matrix.cpu().detach().numpy()[0])
    print("match")
    print(match)
    print("x1")
    print(x1.cpu().detach().numpy())
    print("x2")
    print(x2.cpu().detach().numpy())
    print("x3")
    print(x3.cpu().detach().numpy())
    print("x4")
    print(x4.cpu().detach().numpy())