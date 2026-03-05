import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# def test_img(net_g, datatest, args):
#     net_g.eval()
#     # testing
#     test_loss = 0
#     correct = 0
#     data_loader = DataLoader(datatest, batch_size=args.bs)
#     l = len(data_loader)
#     for idx, (data, target) in enumerate(data_loader):
#         if torch.cuda.is_available() and args.gpu != -1:
#             data, target = data.cuda(args.device), target.cuda(args.device)
#         else:
#             data, target = data.cpu(), target.cpu()
#         log_probs = net_g(data)
#         # sum up batch loss
#         test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#         # get the index of the max log-probability
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

#     test_loss /= len(data_loader.dataset)
#     accuracy = 100.00 * correct / len(data_loader.dataset)
#     return accuracy, test_loss

# def test_img(net_g, datatest, args):
#     net_g.eval()
#     test_loss = 0
#     correct = 0
#     if isinstance(datatest, Dataset):
#         data_loader = DataLoader(datatest, batch_size=args.bs)
#     else:
#         data_loader = datatest
#     # 选择设备
#     device = torch.device(f"{args.device}" if torch.cuda.is_available() and args.gpu!= -1 else "cpu")
#     total_samples = len(data_loader) * args.bs
#     for idx, (data, target) in enumerate(data_loader):
#         # 将数据和目标移动到相应设备
#         data, target = data.to(device), target.to(device)
#         log_probs = net_g(data)
#         # 使用 reduction='mean' 计算平均损失
#         test_loss += F.cross_entropy(log_probs, target, reduction='mean').item()
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

#     test_loss /= len(data_loader)
#     accuracy = 100.00 * correct / total_samples
#     return test_loss, accuracy


# def test_img(net_g, dataloader, args):

#     if isinstance(dataloader, Dataset):
#         dataloader = DataLoader(dataloader, batch_size=args.bs)
#     else:
#         dataloader = dataloader
#     net_g.eval()

#     loss_meter = 0
#     acc_meter = 0
#     runcount = 0
#     device = torch.device(f"{args.device}" if torch.cuda.is_available() and args.gpu!= -1 else "cpu")
#     with torch.no_grad():
#         for load in dataloader:
#             data, target = load[:2]
#             data = data.to(device)
#             target = target.to(device)
        
#             pred = net_g(data)  # test = 4
#             loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss
#             pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
#             acc_meter += pred.eq(target.view_as(pred)).sum().item()
#             runcount += data.size(0) 

#     loss_meter /= runcount
#     acc_meter /= runcount
#     return loss_meter, acc_meter * 100  

def test_img(net_g, dataloader, args):
    # 统一成 DataLoader
    if isinstance(dataloader, Dataset):
        data_loader = DataLoader(dataloader, batch_size=args.bs, shuffle=False)
    else:
        data_loader = dataloader

    net_g.eval()
    device = torch.device(f"{args.device}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:     # ✅ 直接解包，不用 load[:2]
            data = data.to(device)
            target = target.to(device).long()  # ✅ CE 需要 long label
            if target.ndim == 2 and target.size(1) == 1:
                target = target.squeeze(1)
            elif target.ndim > 1:
                target = torch.argmax(target, dim=1)
            logits = net_g(data)
            loss_sum += F.cross_entropy(logits, target, reduction='sum').item()

            pred = logits.argmax(dim=1)      # ✅ shape [N]
            correct += (pred == target).sum().item()
            total += target.size(0)

    avg_loss = loss_sum / total
    acc = 100.0 * correct / total
    return acc, avg_loss