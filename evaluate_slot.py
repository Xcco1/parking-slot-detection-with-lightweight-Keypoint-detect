import numpy as np
import torch.nn as nn
import torch, argparse, pdb, os, math
import cv2
from models.Movenet import MoveNet
from data.struct_ import MarkingPoint
from data.dataset import ParkingSlotDataset
from utils.common import get_predicted_points, calc_precision_recall
from utils.common import calc_average_precision, match_marking_points, match_slots
from utils.common import _nms, _topk, plot_points, _topslot, plot_point_slot
from utils.common import inference_slots_v2


parser = argparse.ArgumentParser()

parser.add_argument('--dataset_directory', type=str, default="E:\ps2.0\ps2.0/")
parser.add_argument('--feature_channel', type=int, default=3, help="The output channel of network")
parser.add_argument("--weight_dir", type=str, default="./experiments/best.pth", help="the dir of the trained model's weight.")
parser.add_argument("--save_dir", type=str, default="./result", help="the dir of the result fold path.")


parser.add_argument("--gpus", type=str, default="0", help="Choose the device of GPU.")

config = parser.parse_args()

if not config.gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus


if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_grad_enabled(False)
    
    #model = DBFace(wide=64, has_ext=True, upmode="UCBA").to(device)
    model=MoveNet().cuda()
    if not config.weight_dir is None:
        model.load_state_dict(torch.load(config.weight_dir))
        print("Loading the weights successfully!")
    
    model.eval()    
    
    val_dataset = ParkingSlotDataset(root=config.dataset_directory, train=False)
    
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    
    groundtruth_list = []
    prediction_list = []
    
    threshold = 0.4
    
    for iter_idx, (image, marking_points, name, slots) in enumerate(val_dataset):
        
        #print(name)
        
        image = torch.unsqueeze(image, 0).to(device)
        
        hm, sl, di= model(image)
        
        #pdb.set_trace()
        
        _, num_classes, hm_height, hm_width = hm.shape
        
        hm = hm[0].reshape(1, num_classes, hm_height, hm_width)
        
        nmskey = _nms(hm, 3)
        
        kscore, kinds, kcls, kys, kxs, ksin, kcos = _topk(nmskey, di)
        
        kys = kys.cpu().data.numpy().astype(np.int)
        kxs = kxs.cpu().data.numpy().astype(np.int)
        ksin = ksin.cpu().data.numpy().astype(np.float64)
        kcos = kcos.cpu().data.numpy().astype(np.float64)
        
        predicted_points = []
        
        temp = set()
        
        for ind in range(kscore.shape[1]):
            score = kscore[0, ind].item()
            if score > threshold:
                yval = kys[0, ind] / 128
                xval = kxs[0, ind] / 128
                sin_value = ksin[0, ind]
                cos_value = kcos[0, ind]
                
                direction = math.atan2(sin_value, cos_value)
                
                pred_point = MarkingPoint(xval, yval, direction)
                if pred_point in temp:
                    continue
                temp.add(pred_point)
                predicted_points.append((float('%.3f' % score), pred_point))
        
#        if config.save_dir is not None:
#            plot_points(config.dataset_directory, predicted_points, name, config.save_dir, marking_points)
        
        ######
        _, num_classes, sl_height, sl_width = sl.shape
        
        sl = sl[0].reshape(1, num_classes, sl_height, sl_width)
        
        slkey = _nms(sl, 3)
        
        kscore, kinds, kcls, kys, kxs = _topslot(slkey)
        
        kys = kys.cpu().data.numpy().astype(np.int)
        kxs = kxs.cpu().data.numpy().astype(np.int)
        
        point_slots = []
        
        temp = set()
        
        for ind in range(kscore.shape[1]):
            score = kscore[0, ind].item()
            if score > threshold:
                yval = kys[0, ind] / 128
                xval = kxs[0, ind] / 128
    
                pred_point = MarkingPoint(xval, yval, 0)
                if pred_point in temp:
                    continue
                temp.add(pred_point)
                point_slots.append([float('%.3f' % score), pred_point])
                
        groundtruth_list.append(slots)
        
        
        pred_slot = inference_slots_v2(point_slots, predicted_points)
        
        prediction_list.append(pred_slot)
        
        if config.save_dir is not None:
            plot_point_slot(config.dataset_directory, slots, name, config.save_dir, point_slots, predicted_points, pred_slot)
        
    precisions, recalls = calc_precision_recall(groundtruth_list, prediction_list, match_slots)
    
    print(np.mean(np.array(precisions)))
    print(np.mean(np.array(recalls)))
    
    
    average_precision = calc_average_precision(precisions, recalls)  #0.7840129983663647
    
    print("Precision is:", average_precision)
        