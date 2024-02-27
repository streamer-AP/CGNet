import torch

def scheduler_builder(args,optimizer):
    print(optimizer)
    if args.type=="step":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)
    elif args.type=="cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max, args.eta_min)
    elif args.type=="multi_step":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)
    elif args.type=="consine_warm_restart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.T_max, args.eta_min)