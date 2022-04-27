import torch, os, math, random, pdb, random
import numpy as np
import torch.nn as nn
import argparse

from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils import data
import torch.optim as optim

from data.dataset import ParkingSlotDataset
from utils.encoding import DataParallelCriterion
from utils.loss import SELoss, FocalLoss, OHEMSELoss, weightedMSE, Swingloss
from models.network import DBFace
from models.Movenet import MoveNet

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_directory', type=str, default="E:\ps2.0\ps2.0/")
parser.add_argument('--input_image_size', type=int, default=512, help="The default image size for network")
parser.add_argument('--feature_map_size', type=int, default=128, help="The output feature size of network")

parser.add_argument('--num_epochs', type=int, default=1000, help="Number of training step")
parser.add_argument('--batch_size', type=int, default=32, help="Number of images sent to the network in one step")
parser.add_argument('--num_workers', type=int, default=0, help="Number of work for dataLoader.")
parser.add_argument("--learning-rate", type=float, default=0.001,
                    help="Base learning rate for training with polynomial decay.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
parser.add_argument("--random-seed", type=int, default=1024, help="Random seed to have reproducible results.")
parser.add_argument("--weight-decay", type=float, default=1e-4, help="Regularisation parameter for L2-loss.")
parser.add_argument("--gpus", type=str, default="0", help="Choose the device of GPU.")

parser.add_argument("--alpha", type=float, default=1.0, help="Set the weight of point heatmap loss.")
parser.add_argument("--gama", type=float, default=1.0, help="Set the weight of slot heatmap loss.")
parser.add_argument("--beta", type=float, default=4.0, help="Set the weight of direction loss.")

parser.add_argument("--restore-from", type=str, default=None, help="Where restore model parameters from.")
parser.add_argument("--snapshot-dir", type=str, default="./experiments", help="Where to save snapshots of the model.")

config = parser.parse_args()

if not config.gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

if __name__ == "__main__":
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    if not os.path.exists(config.snapshot_dir):
        os.makedirs(config.snapshot_dir)

    writer = SummaryWriter(config.snapshot_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    cudnn.enabled = True

    #model = DBFace(wide=64, has_ext=True, upmode="UCBA")
    #model.init_weights()
    model=MoveNet()
    # criterion = SELoss()
    #hm_loss = FocalLoss()
    #hm_loss=weightedMSE()
    hm_loss = Swingloss()
    dr_loss = SELoss()
    # dr_loss = OHEMSELoss()

    if len(config.gpus.split(",")) != 1:
        device_num = [int(x) for x in config.gpus.split(",")]
        model = nn.DataParallel(model, device_ids=device_num)
        criterion = DataParallelCriterion(dr_loss, device_ids=device_num)

    model = model.to(device)
    hm_loss = hm_loss.to(device)
    dr_loss = dr_loss.to(device)

    model.load_state_dict(torch.load("./experiments/best.pth"))
    model.train()
    print('parameters :', sum(x.numel() for x in model.parameters()) / 1e6)
    train_dataset = ParkingSlotDataset(root=config.dataset_directory, size=(512, 512), train=True,
                                       fm_height=config.feature_map_size, fm_width=config.feature_map_size)
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                   num_workers=config.num_workers, pin_memory=True, collate_fn=lambda x: list(zip(*x)))

    optimizer1 = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    schedule1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=5,
                                                           verbose=False, threshold=0.01, threshold_mode='rel',
                                                           cooldown=2, min_lr=0.00008, eps=1e-08)
    schedule1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.6)
    #schedule1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.96,last_epoch=-1)                                                       
    
    optimizer2 = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, dampening=0.5, weight_decay=0.0001,
                           nesterov=False)
    schedule2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=8, gamma=0.6)


    # optimizer1.zero_grad()
    # optimizer2.zero_grad()
    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))


    def adjust_learning_rate(optimizer, i_iter, max_iter):
        lr = lr_poly(config.learning_rate, i_iter, max_iter, config.power)
        optimizer.param_groups[0]['lr'] = lr
        return lr


    one_epoch_max_iter = len(train_loader)
    min_loss = float('inf')

    for epoch in range(config.num_epochs):
        train_epoch_loss = []
        train_hm_loss = []
        train_dr_loss = []
        train_sl_loss = []
        if epoch < 1000:
            print('using Adam')
            optimizer = optimizer1
            schedule = schedule1
        else:
            print('using SGD')
            optimizer = optimizer2
            schedule = schedule2
        for iter_idx, (
        image, heatmap_gt, heatmap_pos_weight, direction_gt, slot_map_gt, slot_map_pos_weight, marking_points, _,
        _) in enumerate(train_loader):
            batch_points = sum([len(marking_point) for marking_point in marking_points])
            if batch_points == 0:
                batch_points == 1

            images = torch.stack(image).to(device)
            heatmap_gt = torch.stack(heatmap_gt).to(device)
            heatmap_pos_weight = torch.stack(heatmap_pos_weight).to(device)
            direction_gt = torch.stack(direction_gt).to(device)
            slot_map_gt = torch.stack(slot_map_gt).to(device)
            slot_map_pos_weight = torch.stack(slot_map_pos_weight).to(device)

            # pdb.set_trace()
            hm, sl, di = model(images)
            hm = torch.clamp(hm, min=1e-4, max=1 - 1e-4)
            sl = torch.clamp(sl, min=1e-4, max=1 - 1e-4)

            loss1 = hm_loss(hm, heatmap_gt) / batch_points
            loss2 = dr_loss(di, direction_gt) / batch_points
            loss3 = hm_loss(sl, slot_map_gt) / batch_points
            loss = config.alpha * loss1 + config.beta * loss2 + config.gama * loss3

            # pdb.set_trace()
            # lr = adjust_learning_rate(optimizer, epoch * one_epoch_max_iter + iter_idx, config.num_epochs * one_epoch_max_iter)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_hm_loss.append(loss1.item())
            train_dr_loss.append(loss2.item())
            train_sl_loss.append(loss3.item())
            train_epoch_loss.append(loss.item())
            lr = optimizer.param_groups[0]['lr']
            if iter_idx % 10 == 0:
                writer.add_scalar("learning_rate", lr, epoch * one_epoch_max_iter + iter_idx + 1)
                writer.add_scalar("training_loss", sum(train_epoch_loss) / len(train_epoch_loss),
                                  epoch * one_epoch_max_iter + iter_idx + 1)
                writer.add_scalar("heatmap_loss", sum(train_hm_loss) / len(train_hm_loss),
                                  epoch * one_epoch_max_iter + iter_idx + 1)
                writer.add_scalar("direction_loss", sum(train_dr_loss) / len(train_dr_loss),
                                  epoch * one_epoch_max_iter + iter_idx + 1)
                writer.add_scalar("slot_de_loss", sum(train_sl_loss) / len(train_sl_loss),
                                  epoch * one_epoch_max_iter + iter_idx + 1)

                print(
                    "Epoch:{} || Iteration:[{}/{}] || Learning Rate:{} || Loss:{:.3f} || point Loss:{:.4f} || Direction Loss:{:.3f} || slot Loss:{:.4f} ".format(
                        epoch, iter_idx, one_epoch_max_iter, lr, sum(train_epoch_loss) / len(train_epoch_loss),
                        sum(train_hm_loss) / len(train_hm_loss), sum(train_dr_loss) / len(train_dr_loss),
                        sum(train_sl_loss) / len(train_sl_loss)))

            if iter_idx == one_epoch_max_iter - 1:
                # print("Save the state dict of model in {} epoch".format(epoch))
                # torch.save(model.state_dict(), os.path.join(config.snapshot_dir, "parkingslot_epoch"+str(epoch)+".pth"))
                current_loss = sum(train_epoch_loss) / len(train_epoch_loss)
                if current_loss < min_loss:
                    print("Save the best state dict of model!")
                    torch.save(model.state_dict(), os.path.join(config.snapshot_dir, "best.pth"))
                    min_loss = current_loss
        schedule.step(current_loss)
