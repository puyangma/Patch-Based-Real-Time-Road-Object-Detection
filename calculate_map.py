import json
import re
import os
import numpy as np
from utils.utils import *
import sys

def format_annotation(annotation, w=1280, h=720):
    x1 = w * (annotation[1] - annotation[3]/2)
    y1 = h * (annotation[2] - annotation[4]/2)
    x2 = w * (annotation[1] + annotation[3]/2)
    y2 = h * (annotation[2] + annotation[4]/2)
    
    return [x1, y1, x2, y2]

def calculate_map(classes, detections_dict, annotations_dict, iou_thres=0.4):
    average_precisions = {}
    for label in classes:
        true_positives = []
        scores = []
        num_annotations = 0

        for image_file_name in detections_dict:
            annotation_file_name = image_file_name.replace('.png', '.txt').replace('.jpg', '.txt')
            annotations = np.asarray(annotations_dict[annotation_file_name][label])
            num_annotations += annotations.shape[0]
            detected_annotations = []
            for detection in detections_dict[image_file_name][label]:
                x1 = detection["box2d"]["x1"]
                y1 = detection["box2d"]["y1"]
                x2 = detection["box2d"]["x1"] + detection["box2d"]["box_w"]
                y2 = detection["box2d"]["y1"] + detection["box2d"]["box_h"] 
                bbox = [x1, y1, x2, y2]
                score = detection["confidence"]
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]
                
                if max_overlap >= iou_thres and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        # no annotations -> AP for this class is 0                                                                                                     
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        
        # sort by score                                                                                                                                
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives                                                                                                   
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision                                                                                                                 

        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision                                                                                                                    
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    print("Average Precisions:")
    for c, ap in average_precisions.items():
        print(f"+ Class '{c}' - AP: {ap}")
    
    mAP = np.mean(list(average_precisions.values()))
    print(f"mAP: {mAP}")

def process_detections_json(detections_json_file_name, classes, labels_dir):
    with open(detections_json_file_name, 'r') as f:
        detections_dict = {}
        annotations_dict = {}
        loaded_json = json.load(f)
        for crop in loaded_json:
            match = re.match("(.*)_[0-9]+_[0-9]+\.(jpg|gif|png|bmp)", crop["name"])
            if match:
                image_file_name = match.group(1) + "." + match.group(2)
                if image_file_name not in detections_dict:
                    detections_dict[image_file_name] = {}
                    for class_name in classes: detections_dict[image_file_name][class_name] = []

                for label in crop["labels"]:
                    detections_dict[image_file_name][label["category"]].append(label)

                annotation_file_name = image_file_name.replace('.png', '.txt').replace('.jpg', '.txt')
                if annotation_file_name not in annotations_dict:
                    annotations_dict[annotation_file_name] = {}
                    for class_name in classes: annotations_dict[annotation_file_name][class_name] = []
                    if os.path.exists(labels_dir + annotation_file_name):
                        annotations = np.loadtxt(labels_dir + annotation_file_name).reshape(-1, 5)
                        for annotation in annotations:
                            class_name = classes[int(annotation[0])]
                            annotations_dict[annotation_file_name][class_name].append(format_annotation(annotation))

    return detections_dict, annotations_dict

def main(argv):
    detections_json_file_name = argv[1]
    classes = load_classes(argv[2]) # Extracts class labels from file         
    labels_dir = argv[3]                                                                       
    detections_dict, annotations_dict = process_detections_json(detections_json_file_name, classes, labels_dir)    
    calculate_map(classes, detections_dict, annotations_dict)

# Example usage: python3 calculate_map.py detect_patchcrd.json data/bdd.names labels/                                         
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("ERROR: Invalid input arguments.")
    else:
        main(sys.argv)
