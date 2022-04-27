import json, os, cv2, sys, pdb, random, torch, math
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import albumentations as A
import numpy as np
import torchvision.transforms as tf
sys.path.append("../")

from data.struct_ import MarkingPoint, Slot
from data.transform import *
from scipy.ndimage import gaussian_filter
class ParkingSlotDataset(Dataset):
    def __init__(self, root, size=(512, 512), train=True, fm_height=128, fm_width=128):
        super(ParkingSlotDataset, self).__init__()
        self.root = root
        self.image_size = size
        self.sample_names = []
        self.image_transform = ToTensor()
        self.augment=A.Compose([
        #A.MotionBlur(p=0.2)
        A.Blur(blur_limit=(5,5),p=0.5),
        #A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Cutout(num_holes=30, max_h_size=30, max_w_size=30, fill_value=64, p=0.5)
        ],p=0.5)
        self.train = train
        self.fm_height = fm_height
        self.fm_width = fm_width
        
        self.classes = 0
        self.h_radius, self.w_radius, self.posweight_radius = 2, 2, 2
        
        self.angle_list = [x for x in range(0, 360, 10)]
        
        if train:
            with open(os.path.join(root, "split", "train.txt"), "r") as reader:
                for line in reader:
                    self.sample_names.append(line)
        else:
            with open(os.path.join(root, "split", "test.txt"), "r") as reader:
                for line in reader:
                    self.sample_names.append(line)
    
    def get_slots(self, slots, marks):
        "Read label to get ground truth slot"
        
        slots = np.array(slots)
        marks = np.array(marks)
        
        if slots.size == 0:
            return []
        if len(slots.shape) < 2:
            slots = np.expand_dims(slots, axis=0)
        
        if len(marks.shape) < 2:
            marks = np.expand_dims(marks, axis=0)
        ground_truths = []
        for slot in slots:
            mark_a = marks[slot[0]-1]
            mark_b = marks[slot[1]-1]
            coords = np.array([mark_a[0], mark_a[1], mark_b[0], mark_b[1]])
            ground_truths.append(Slot(*coords))
        return ground_truths
    
    def __getitem__(self, index):
        image_path, label_path = self.sample_names[index].split()
        
        image = cv2.imread(os.path.join(self.root, image_path))

        image = cv2.resize(image, self.image_size)
        
        with open(os.path.join(self.root, label_path), "r") as file:
            label = json.load(file)
        
        centralied_marks=np.array(label['marks'], dtype=np.float64)
        if len(centralied_marks.shape) < 2:
            centralied_marks = np.expand_dims(centralied_marks, axis=0)
        centralied_marks[:, 0:4] -= 300.5
        
        angle = random.choice(self.angle_list)
        rotated_marks = rotate_centralized_marks(centralied_marks, angle)
        if self.train and boundary_check(rotated_marks) and overlap_check(rotated_marks):
            image = self.augment(image=np.array(image))
            image=image['image']
            image =rotate_image(image, angle)
            centralied_marks = generalize_marks(rotated_marks)
        else:
            centralied_marks = generalize_marks(centralied_marks)
        
        marking_points = []
        for point in centralied_marks:
            marking_points.append(MarkingPoint(*point[:3]))
        #image=tf.RandomErasing(p=0.5,scale=(0.02,0.33),ratio=(0.3,3.3),value=(32,32,32))(image)
        image=self.image_transform(image)
        #image=tf.ColorJitter(brightness=0.5, contrast=0.5,saturation=0.5,hue=0.5)(image) #added in 2022/2/10
        slots = self.get_slots(label["slots"], marking_points)        
        
        if not self.train:
            return image, marking_points, image_path, slots

        heatmap_gt = np.zeros((1, self.fm_height, self.fm_width), np.float32)
        heatmap_pos_weight = np.zeros((1, self.fm_height, self.fm_width), np.float32)
        #direction
        direction_gt = np.full_like(np.zeros((2, self.fm_height, self.fm_width)), 255, dtype=np.float32)

        for point in marking_points:
            cx = math.floor(point.x * self.fm_height)
            cy = math.floor(point.y * self.fm_width)
            _ = draw_truncate_gaussian(heatmap_gt[self.classes, :, :], (cx, cy), self.h_radius, self.w_radius)
            draw_gaussian(heatmap_pos_weight[self.classes, :, :], (cx, cy), self.posweight_radius)
            draw_direction_matrix(direction_gt, (cx, cy), point.direction)
        
        slot_map_gt = np.zeros((1, self.fm_height, self.fm_width), np.float32)
        slot_map_pos_weight = np.zeros((1, self.fm_height, self.fm_width), np.float32)
        
        for slot in slots:
            cx = (slot.x1 + slot.x2) / 2
            cy = (slot.y1 + slot.y2) / 2
            cx = math.floor(cx * self.fm_height)
            cy = math.floor(cy * self.fm_width)
            _ = draw_truncate_gaussian(slot_map_gt[self.classes, :, :], (cx, cy), self.h_radius, self.w_radius)
            draw_gaussian(slot_map_pos_weight[self.classes, :, :], (cx, cy), self.posweight_radius)
        vector_heatmap = np.zeros((len(marking_points), 128, 128))
        heatmap = np.zeros((128, 128))
        for i in range(len(marking_points)):
            point=marking_points[i]
            p0_x = int(round(512 * point.x)/4)
            p0_y = int(round(512 * point.y)/4)
            cos_val = math.cos(point.direction)
            sin_val = math.sin(point.direction)
            p1_x = int(round(p0_x + 50 * cos_val)/4)
            p1_y = int(round(p0_y + 50 * sin_val)/4)
            pxheatmap=int(round(p0_x + 1.5* cos_val))
            pyheatmap=int(round(p0_y + 1.5* sin_val))
            cv2.line(vector_heatmap[i], (p0_x, p0_y), (pxheatmap, pyheatmap), color=1,thickness=1)
            vector_heatmap[i] = gaussian_filter(vector_heatmap[i],1)

            maxi = np.amax(vector_heatmap[i])
            vector_heatmap[i] = vector_heatmap[i] / maxi
            heatmap += vector_heatmap[i]

        cv2.imwrite('hp.jpg',heatmap*255)
        cv2.imwrite('hpo.jpg',heatmap_pos_weight[0]*255)
        heatmap=np.expand_dims(heatmap,axis=0)

        return image, torch.from_numpy(heatmap_gt), torch.from_numpy(heatmap), torch.from_numpy(direction_gt), torch.from_numpy(slot_map_gt), torch.from_numpy(slot_map_pos_weight), marking_points, image_path, slots
        
        
    
    def __len__(self):
        return len(self.sample_names)
        
if __name__ == "__main__":
    dataset = ParkingSlotDataset(root=r"E:\ps2.0\ps2.0/", train=True)
    #pdb.set_trace()
    #dataset.__getitem__(0)
    if not os.path.exists("./result"):
        os.makedirs("./result")
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
    for index in range(len(dataset)):
        image, _, _, _, _, _, marking_points, image_path, slots = dataset.__getitem__(index)
        print(image)
        #pdb.set_trace()
        #print(image_path)
        name = image_path.split("/")[-1]
        
        #print(name)
        
        for point in marking_points:
            p0_x = int(round(512 * point.x))
            p0_y = int(round(512 * point.y))
            cos_val = math.cos(point.direction)
            sin_val = math.sin(point.direction)
            p1_x = int(round(p0_x + 50*cos_val))
            p1_y = int(round(p0_y + 50*sin_val))
            cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 5)
            #cv2.circle(image, (p0_x, p0_y), 5, (0, 0, 255), -1)
        for index, slot in enumerate(slots):
            p0_x = int(round(512 * slot.x1))
            p0_y = int(round(512 * slot.y1))
            p1_x = int(round(512 * slot.x2))
            p1_y = int(round(512 * slot.y2))
            cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), colors[index%3], 5)
        cv2.imwrite("./result/"+name, image)
        #break
        
        
        
        
    