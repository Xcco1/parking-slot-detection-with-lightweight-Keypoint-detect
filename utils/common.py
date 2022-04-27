import torch, cv2, pdb, os, math
import numpy as np
import torch.nn as nn
from data.struct_ import MarkingPoint, Slot



COLORS = [(0, 0, 0),(128, 64, 128),(244, 35, 232),(70, 70, 70),(102, 102, 156),(190, 153, 153),(244, 35, 232)]

def inference_slots(marking_points):
    """Inference slots based on marking points."""
    num_detected = len(marking_points)
    slots = set()
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            point_i = marking_points[i]
            point_j = marking_points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            #pdb.set_trace()
            #if not (0.010771278151623496 <= distance <= 0.1099427457599304):
            if not (60 <= distance <= 180): #60
                continue
            # step 2: check the direction with two points
            if not (-0.08 <=point_i.direction - point_j.direction <= 0.08):
                continue
#            if point_i.direction * point_j.direction > 0:
#                if not (-0.08 <=point_i.direction - point_j.direction <= 0.08):
#                    continue
#            else:
#                if not (-0.08 <=abs(point_i.direction) - abs(point_j.direction) <= 0.08):
#                    continue
            # Step 3: pass through filtration.
            if pass_through_third_point(marking_points, i, j):
                continue
            
            slots.add(Slot(*[point_i.x, point_i.y, point_j.x, point_j.y]))
#            pdb.set_trace()
#            result = pair_marking_points(point_i, point_j)
#            if result == 1:
#                slots.append((i, j))
#            elif result == -1:
#                slots.append((j, i))
    return list(slots)

def inference_slots_v2(point_slots, marking_points):
    """Inference slots based on marking points."""
    num_detected = len(marking_points)
    slots = set()
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            point_i = marking_points[i]
            point_j = marking_points[j]
#            # Step 1: length filtration.
#            distance = calc_point_squre_dist(point_i, point_j)
#            if not (60 <= distance <= 180): #60
#                continue
#            # step 2: check the direction with two points
#            if not (-0.08 <=point_i[1].direction - point_j[1].direction <= 0.08):
#                continue
            # step 3
            if not pass_through_slot_point(point_slots, point_i, point_j):
                continue
            if pass_through_third_point(marking_points, i, j):
                continue
            
            slots.add(Slot(*[point_i[1].x, point_i[1].y, point_j[1].x, point_j[1].y]))

    return list(slots)


def pass_through_slot_point(point_slots, point_i, point_j, SLOT_SUPPRESSION_DOT_PRODUCT_THRESH=0.8):
    """See whether the line between two points pass through a third point."""
    x_1 = point_i[1].x
    y_1 = point_i[1].y
    x_2 = point_j[1].x
    y_2 = point_j[1].y
    point_0 = MarkingPoint(*[(x_1+x_2) /2, (y_1+y_2) / 2, 0])
    
    for point_idx, point in enumerate(point_slots):
        distance = calc_point_squre_dist(point[1], point_0)
        if distance < 6:
            return True
    return False



def pass_through_third_point(marking_points, i, j, SLOT_SUPPRESSION_DOT_PRODUCT_THRESH=0.8):
    """See whether the line between two points pass through a third point."""
    x_1 = marking_points[i][1].x
    y_1 = marking_points[i][1].y
    x_2 = marking_points[j][1].x
    y_2 = marking_points[j][1].y
    for point_idx, point in enumerate(marking_points):
        if point_idx == i or point_idx == j:
            continue
        x_0 = point[1].x
        y_0 = point[1].y
        vec1 = np.array([x_0 - x_1, y_0 - y_1])
        vec2 = np.array([x_2 - x_0, y_2 - y_0])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        if np.dot(vec1, vec2) > SLOT_SUPPRESSION_DOT_PRODUCT_THRESH:
            return True
    return False




def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, direction, K=60):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    
    topk_cos = direction[:, 0].view(batch, -1)[0][topk_inds]
    topk_sin = direction[:, 1].view(batch, -1)[0][topk_inds]
    
    topk_clses = torch.true_divide(topk_inds, (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = torch.true_divide(topk_inds, width).float()
    topk_xs   = (topk_inds % width).float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs, topk_sin, topk_cos


def _topslot(scores, K=60):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    
    topk_clses = torch.true_divide(topk_inds, (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = torch.true_divide(topk_inds, width).float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs



def plot_points(image_path, pred_points, image_name, save_path, marking_points=None):
    image = cv2.imread(os.path.join(image_path, image_name))
    name = image_name.split("/")[-1]
    width, height, _ = image.shape
    for confidence, pred_point in pred_points:
        p0_x = int(round(width * pred_point.x - 0.5))
        p0_y = int(round(height * pred_point.y - 0.5))
        
        cos_val = math.cos(pred_point.direction)
        sin_val = math.sin(pred_point.direction)
        p1_x = int(round(p0_x + 50*cos_val))
        p1_y = int(round(p0_y + 50*sin_val))
        cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 5)
        cv2.circle(image, (p0_x, p0_y), 5, (0, 0, 255), -1)
        #cv2.putText(image, str(confidence), (p0_x, p0_y),
                   #cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    if marking_points is not None:
        for label_point in marking_points:
            p0_x = int(round(width * label_point.x - 0.5))
            p0_y = int(round(height * label_point.y - 0.5))
            #if abs(label_point.direction) > math.pi / 2: 
            cos_val = math.cos(label_point.direction)
            sin_val = math.sin(label_point.direction)
            p1_x = int(round(p0_x + 50*cos_val))
            p1_y = int(round(p0_y + 50*sin_val))
            cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 5)
            cv2.circle(image, (p0_x, p0_y), 5, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(save_path, name), image)
    

def plot_point_slot(image_path, slots, image_name, save_path, point_slots, predicted_points, pred_slots):
    image = cv2.imread(os.path.join(image_path, image_name))
    name = image_name.split("/")[-1]
    width, height, _ = image.shape
    for index, slot in enumerate(slots):
        p0_x = int(round(width * slot.x1))
        p0_y = int(round(height * slot.y1))
        p1_x = int(round(width * slot.x2))
        p1_y = int(round(height * slot.y2))
        cx = int(round((p0_x + p1_x) / 2))
        cy = int(round((p0_y + p1_y) / 2))
        #cv2.circle(image, (cx, cy), 5, (255, 255, 255), 4)
        
    for confidence, pred_point in point_slots:
        #pdb.set_trace()
        p0_x = int(round(width * pred_point.x - 0.5))
        p0_y = int(round(height * pred_point.y - 0.5))
        cv2.circle(image, (p0_x, p0_y), 3, (128, 128, 128), 3)
        #cv2.putText(image, str(confidence), (p0_x, p0_y),
                   #cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    
    for confidence, pred_point in predicted_points:
        p0_x = int(round(width * pred_point.x - 0.5))
        p0_y = int(round(height * pred_point.y - 0.5))
        
        cos_val = math.cos(pred_point.direction)
        sin_val = math.sin(pred_point.direction)
        p1_x = int(round(p0_x + 50*cos_val))
        p1_y = int(round(p0_y + 50*sin_val))
        cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 255, 1), 2)
        #cv2.circle(image, (p0_x, p0_y), 3, (0, 0, 255), -1)
        #cv2.putText(image, str(confidence), (p0_x, p0_y),
                   #cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
                   
    for index, slot in enumerate(pred_slots):
        p0_x = int(round(width * slot.x1))
        p0_y = int(round(height * slot.y1))
        p1_x = int(round(width * slot.x2))
        p1_y = int(round(height * slot.y2))
        cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (255,0,0), 2)
                   
    cv2.imwrite(os.path.join(save_path, name), image)

def plot_slots(image_path, slots, image_name, save_path, pred_points, plot_point = True):
    image = cv2.imread(os.path.join(image_path, image_name))
    name = image_name.split("/")[-1]
    width, height, _ = image.shape
    for index, slot in enumerate(slots):
        p0_x = int(round(width * slot.x1))
        p0_y = int(round(height * slot.y1))
        p1_x = int(round(width * slot.x2))
        p1_y = int(round(height * slot.y2))
        cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), COLORS[index%7], 5)
        
    if plot_point == True:
        for confidence, pred_point in pred_points:
            p0_x = int(round(width * pred_point.x - 0.5))
            p0_y = int(round(height * pred_point.y - 0.5))
            
            cos_val = math.cos(pred_point.direction)
            sin_val = math.sin(pred_point.direction)
            p1_x = int(round(p0_x + 50*cos_val))
            p1_y = int(round(p0_y + 50*sin_val))
            cv2.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 5)
            cv2.circle(image, (p0_x, p0_y), 5, (255, 255, 255), 4)
            cv2.putText(image, str(confidence), (p0_x, p0_y),
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    cv2.imwrite(os.path.join(save_path, name), image)



def non_maximum_suppression(pred_points):
    """Perform non-maxmum suppression on marking points."""
    suppressed = [False] * len(pred_points)
    for i in range(len(pred_points) - 1):
        for j in range(i + 1, len(pred_points)):
            i_x = pred_points[i][1].x
            i_y = pred_points[i][1].y
            j_x = pred_points[j][1].x
            j_y = pred_points[j][1].y
            # 0.0625 = 1 / 16
            if abs(j_x - i_x) < 0.0078125 and abs(j_y - i_y) < 0.0078125:
                idx = i if pred_points[i][0] < pred_points[j][0] else j
                suppressed[idx] = True
    if any(suppressed):
        unsupres_pred_points = []
        for i, supres in enumerate(suppressed):
            if not supres:
                unsupres_pred_points.append(pred_points[i])
        return unsupres_pred_points
    return pred_points

def get_predicted_points(prediction, thresh, BOUNDARY_THRESH=0.05):
    """Get marking points from one predicted feature map."""
    assert isinstance(prediction, torch.Tensor)
    predicted_points = []
    prediction = prediction.detach().cpu().numpy()
    for i in range(prediction.shape[1]):
        for j in range(prediction.shape[2]):
            if prediction[0, i, j] >= thresh:
                xval = j
                yval = i
                xval = (j + prediction[1, i, j]) / prediction.shape[2]
                yval = (i + prediction[2, i, j]) / prediction.shape[1]
#                if not (BOUNDARY_THRESH <= xval <= 1-BOUNDARY_THRESH
#                        and BOUNDARY_THRESH <= yval <= 1-BOUNDARY_THRESH):
#                    continue
                marking_point = MarkingPoint(
                    xval, yval)
                predicted_points.append((prediction[0, i, j], marking_point))
    #return non_maximum_suppression(predicted_points)
    return predicted_points
    

def calc_point_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a.x - point_b.x
    disty = point_a.y - point_b.y
    #return distx ** 2 + disty ** 2
    return 512 * math.sqrt(distx ** 2 + disty ** 2)

def calc_squre_dist(point_a, point_b):
    """Calculate distance between two marking points."""
    distx = point_a.x - point_b.x
    disty = point_a.y - point_b.y
    return distx ** 2 + disty ** 2
    

def match_marking_points(point_a, point_b, SQUARED_DISTANCE_THRESH=0.000277778):
    """Determine whether a detected point match ground truth."""
    dist_square = calc_squre_dist(point_a, point_b[1])
#    angle = calc_point_direction_angle(point_a, point_b)
#    if point_a.shape > 0.5 and point_b.shape < 0.5:
#        return False
#    if point_a.shape < 0.5 and point_b.shape > 0.5:
#        return False
#    return (dist_square < config.SQUARED_DISTANCE_THRESH and angle < config.DIRECTION_ANGLE_THRESH)
    return dist_square < SQUARED_DISTANCE_THRESH

def match_slots(slot_a, slot_b, SQUARED_DISTANCE_THRESH=0.000277778):
    """Determine whether a detected slot match ground truth."""
    
    dist_x1 = slot_b.x1 - slot_a.x1
    dist_y1 = slot_b.y1 - slot_a.y1
    squared_dist1 = dist_x1**2 + dist_y1**2
    dist_x2 = slot_b.x2 - slot_a.x2
    dist_y2 = slot_b.y2 - slot_a.y2
    squared_dist2 = dist_x2 ** 2 + dist_y2 ** 2
    
    dist_x3 = slot_b.x2 - slot_a.x1
    dist_y3 = slot_b.y2 - slot_a.y1
    squared_dist3 = dist_x3**2 + dist_y3**2
    
    dist_x4 = slot_b.x1 - slot_a.x2
    dist_y4 = slot_b.y1 - slot_a.y2
    squared_dist4 = dist_x4 ** 2 + dist_y4 ** 2
    
    return (squared_dist1 < SQUARED_DISTANCE_THRESH
            and squared_dist2 < SQUARED_DISTANCE_THRESH) or (squared_dist3 < SQUARED_DISTANCE_THRESH
            and squared_dist4 < SQUARED_DISTANCE_THRESH)

"""Universal procedure of calculating precision and recall."""
import bisect


def match_gt_with_preds(ground_truth, predictions, match_labels):
    """Match a ground truth with every predictions and return matched index."""
    matched_idx = -1
    
    for i, pred in enumerate(predictions):
        #if match_labels(ground_truth, pred[1]) and max_confidence < pred[0]:
        if match_labels(ground_truth, pred):
            matched_idx = i
    return matched_idx


def get_confidence_list(ground_truths_list, predictions_list, match_labels):
    """Generate a list of confidence of true positives and false positives."""
    assert len(ground_truths_list) == len(predictions_list)
    true_positive_list = []
    false_positive_list = []
    num_samples = len(ground_truths_list)
    for i in range(num_samples):
        ground_truths = ground_truths_list[i]
        predictions = predictions_list[i]
        prediction_matched = [False] * len(predictions)
        
        for ground_truth in ground_truths:
            idx = match_gt_with_preds(ground_truth, predictions, match_labels)
            if idx >= 0:
                prediction_matched[idx] = True
                true_positive_list.append(predictions[idx][0])
            else:
                true_positive_list.append(.0)
        for idx, pred_matched in enumerate(prediction_matched):
            if not pred_matched:
                false_positive_list.append(predictions[idx][0])
    return true_positive_list, false_positive_list


def calc_precision_recall(ground_truths_list, predictions_list, match_labels):
    """Adjust threshold to get mutiple precision recall sample."""
    true_positive_list, false_positive_list = get_confidence_list(
        ground_truths_list, predictions_list, match_labels)
    true_positive_list = sorted(true_positive_list)
    false_positive_list = sorted(false_positive_list)
    thresholds = sorted(list(set(true_positive_list)))
    recalls = [0.]
    precisions = [0.]
    for thresh in reversed(thresholds):
        if thresh == 0.:
            recalls.append(1.)
            precisions.append(0.)
            break
        false_negatives = bisect.bisect_left(true_positive_list, thresh)
        true_positives = len(true_positive_list) - false_negatives
        true_negatives = bisect.bisect_left(false_positive_list, thresh)
        false_positives = len(false_positive_list) - true_negatives
        recalls.append(true_positives / (true_positives+false_negatives))
        precisions.append(true_positives / (true_positives + false_positives))
    return precisions, recalls


def calc_average_precision(precisions, recalls):
    """Calculate average precision defined in VOC contest."""
    total_precision = 0.
    for i in range(11):
        index = next(conf[0] for conf in enumerate(recalls) if conf[1] >= i/10)
        total_precision += max(precisions[index:])
    return total_precision / 11