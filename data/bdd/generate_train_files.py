import json
import os
import sys

def get_label_ids(category):
    label_dict = {'traffic light':0, 'traffic sign':1,'person':2,'car':3}
    label_id = label_dict[category]
    
    return label_id
    
    
def rescale_boxes(boxes):
    rescale_width = 416/1280
    rescale_height = 416/1280
    x1 = boxes[0]*rescale_width
    y1 = boxes[1]*rescale_height
    x2 = boxes[2]*rescale_width
    y2 = boxes[3]*rescale_height
    
    return [x1, y1, x2, y2]


def format_boxes(boxes):
    
    x1 = boxes[0]
    y1 = boxes[1]
    x2 = boxes[2]
    y2 = boxes[3]
    width = (x2 - x1)
    height = y2 - y1
    x1 = x1 + (width/2)
    y1 = y1 + (height/2)
    return([(x1/1280), (y1/720), (width/1280), (height/720)])


x = input('Generate files for train or val: ' )

if x == 'train':
    path_to_images = '/images/train2014/'
    json_path = 'bdd100k/labels/bdd100k_labels_images_train.json'
    annotation_path = 'labels/train2014/'
    im_paths = 'trainvalno5k.txt'
else:
    path_to_images = '/images/val2014/'
    json_path = 'bdd100k/labels/bdd100k_labels_images_val.json'
    annotation_path = 'labels/val2014/'
    im_paths = '5k.txt'

json_file = open(json_path)
data = json.load(json_file)
categories_L = []
bounding_boxes_L = []
im_name = []
L = 2000


#convert bounding boxes from BDD into format for training
for i in range(0, L):
    name = data[i]['name']
    im_name.append(name[:-4])
    bounding_boxes = []
    categories = []
    for label in data[i]['labels']:
        if label['category'] == 'traffic light' or label['category'] == 'traffic sign' or label['category'] == 'person' or label['category'] == 'car':
            categories.append(get_label_ids(label['category']))
            x1 = label['box2d']['x1']
            y1 = label['box2d']['y1']
            x2 = label['box2d']['x2']
            y2 = label['box2d']['y2']
            bounding_boxes.append(format_boxes([x1, y1, x2, y2]))
              
    bounding_boxes_L.append(bounding_boxes)
    categories_L.append(categories)
                
        
   

#generate one txt file per image with annotation info for training 
for i in range(0, L):
    f= open(annotation_path + im_name[i] + '.txt',"w+")
    for j in range(0, len(categories_L[i])):
        f.write(str(categories_L[i][j]) + ' ' + str(bounding_boxes_L[i][j][0]) + ' ' +
        str(bounding_boxes_L[i][j][1]) + ' ' + str(bounding_boxes_L[i][j][2]) + ' ' +
        str(bounding_boxes_L[i][j][3]) + '\n')

    f.close()

#generate training file with paths to all the images    
pathname = os.path.dirname(sys.argv[0])
dir_path = os.path.abspath(pathname)

g = open(im_paths, "w+")  
for i in range(0,L):
    g.write(dir_path + path_to_images + im_name[i] +'.jpg' +'\n') 

g.close()  
