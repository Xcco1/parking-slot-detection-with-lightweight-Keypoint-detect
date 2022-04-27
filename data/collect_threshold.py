"""Collect the value range of different propertity of ps dataset."""
import argparse
import json
import math
import os, sys, pdb
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")

from data.struct_ import MarkingPoint
from utils.common import calc_point_squre_dist
from data.transform import generalize_marks


def get_parser():
    """Return argument parser for collecting thresholds."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_directory', default="/mnt/HD/LM-SV/crop_json",
                        help="The location of label directory.")
    return parser


def collect_thresholds(args):
    """Collect range of value from ground truth to determine threshold."""
    distances = []
    direction = []
    direction_diff = []
    cosine = []
    sine = []

    for label_file in os.listdir(args.label_directory):
        print(label_file)
        with open(os.path.join(args.label_directory, label_file), 'r') as file:
            label = json.load(file)
        marks = np.array(label['marks'], dtype=np.float64)
        slots = np.array(label['slots'], dtype=np.int)
        if slots.size == 0:
            continue
        if len(marks.shape) < 2:
            marks = np.expand_dims(marks, axis=0)
        if len(slots.shape) < 2:
            slots = np.expand_dims(slots, axis=0)
        marks[:, 0:4] -= 300.5
        marks = [MarkingPoint(*mark[:3]) for mark in generalize_marks(marks)]
        for slot in slots:
            mark_a = marks[slot[0]]
            mark_b = marks[slot[1]]
            
            x_1 = mark_a.x
            y_1 = mark_a.y
            x_2 = mark_b.x
            y_2 = mark_b.y
            
            x_0 = (x_1 + x_2) / 2
            y_0 = (y_1 + y_2) / 2
            vec1 = np.array([x_0 - x_1, y_0 - y_1])
            vec2 = np.array([x_2 - x_0, y_2 - y_0])
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            
            
            
            
            
            distances.append(calc_point_squre_dist(mark_a, mark_b))
#            cosine.append(math.cos(mark_a.direction))
#            cosine.append(math.cos(mark_b.direction))
#            sine.append(math.sin(mark_a.direction))
#            sine.append(math.sin(mark_b.direction))
#            direction.append(mark_a.direction)
#            direction.append(mark_b.direction)
            direction_diff.append(mark_a.direction - mark_b.direction)

#            vector_ab = np.array([mark_b.x - mark_a.x, mark_b.y - mark_a.y])
#            vector_ab = vector_ab / np.linalg.norm(vector_ab)
#            ab_bridge_direction = math.atan2(vector_ab[1], vector_ab[0])
#            ba_bridge_direction = math.atan2(-vector_ab[1], -vector_ab[0])
#            separator_direction = math.atan2(-vector_ab[0], vector_ab[1])
#
#            sangle = direction_diff(separator_direction, mark_a.direction)
#            if mark_a.shape > 0.5:
#                separator_angles.append(sangle)
#            else:
#                bangle = direction_diff(ab_bridge_direction, mark_a.direction)
#                if sangle < bangle:
#                    separator_angles.append(sangle)
#                else:
#                    bridge_angles.append(bangle)
#
#            bangle = direction_diff(ba_bridge_direction, mark_b.direction)
#            if mark_b.shape > 0.5:
#                bridge_angles.append(bangle)
#            else:
#                sangle = direction_diff(separator_direction, mark_b.direction)
#                if sangle < bangle:
#                    separator_angles.append(sangle)
#                else:
#                    bridge_angles.append(bangle)
    direction_diff = sorted(direction_diff)
    distances = sorted(distances)
    pdb.set_trace()
#    cosine = sorted(cosine)
#    sine = sorted(sine)
    
    plt.figure()
    plt.hist(distances, len(distances) // 10)
    #plt.xlim((0, 0.2))
    #my_x_ticks = np.arange(0, 0.2, 0.02)
    #plt.xticks(my_x_ticks)
    plt.savefig("./distances.png")
    
    plt.figure()
    plt.hist(direction_diff, len(direction_diff) // 10)
    plt.xlim((-0.5, 0.5))
    my_x_ticks = np.arange(-0.5, 0.5, 0.1)
    plt.xticks(my_x_ticks)
    plt.savefig("./direction_diff.png")
    
#    
#    plt.figure()
#    plt.hist(cosine, len(cosine) // 10)
#    #plt.xlim((0, 1.1))
#    #my_x_ticks = np.arange(0, 1.0, 0.1)
#    #plt.xticks(my_x_ticks)
#    plt.ylim((0, 500))
#    my_y_ticks = np.arange(0, 600, 100)
#    plt.yticks(my_y_ticks)
#    
#    plt.savefig("./consine.png")
#    
#    plt.figure()
#    plt.hist(sine, len(sine) // 10)
#    #plt.xlim((0, 1.1))
#    #my_x_ticks = np.arange(0, 1.0, 0.1)
#    #plt.xticks(my_x_ticks)
#    plt.ylim((0, 500))
#    my_y_ticks = np.arange(0, 600, 100)
#    plt.yticks(my_y_ticks)
#    plt.savefig("./sine.png")
#    



if __name__ == '__main__':
    collect_thresholds(get_parser().parse_args())
