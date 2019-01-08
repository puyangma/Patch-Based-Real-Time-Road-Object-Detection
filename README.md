# Patch-Based Real Time Road Object Detection using YOLOv3
codebase modified from PyTorch-YOLOv3: Minimal implementation of YOLOv3 in PyTorch. (https://github.com/eriklindernoren/PyTorch-YOLOv3)

## Table of Contents
- [PyTorch-YOLOv3](#pytorch-yolov3)
  * [Table of Contents](#table-of-contents)
  * [Paper](#paper)
  * [Installation](#installation)
  * [Modifications Made to Existing Repo](#modifications-made-to-existing-repo)
  * [Test](#test)
  * [Train](#train)
  * [Credit](#credit)

## Paper
### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Original Implementation]](https://github.com/pjreddie/darknet)

## Installation
    $ git clone https://github.com/puyangma/Patch-Based-Real-Time-Road-Object-Detection
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Download Berkeley Deep Drive dataset
    go to (http://bdd-data.berkeley.edu/) to download dataset
    some example images are in data/bdd/example_full_size_images
    download in data/bdd
    

## Modifications Made to Existing Repo (authors: Puyang Ma, Naveen Krishnamurthi, Megan Rowe)
    1.In the config folder, we wrote a modified version of yolov3.cfg named yolo-obj.cfg. In this script, we changed classes to the     number of BDD classes (4), in each of the 3 [yolo] layers, changed filters = (classes + 5) x 3 = 27 in each of the three [convolutional] layers before the [yolo] layers, and changed line batch = 64 and subdivisions = 8 for training.
    2.Created file bdd.names in the data folder with objects names each in new line
    3.Created file obj.data in the config folder containing the number of classes, paths to the labels for train and validation sets, path to the bdd.names file, and evaluation metric.
    4.To train the baseline model on full size images, we created a script in the data/bdd folder called generate_train_files.py to convert the BDD dataset into the training format expected by the YOLOv3 implementation. Running this script parses the image labels from the BDD dataset json and creates a .txt file corresponding to each image and writes to the file: <object-class> <x> <y> <width> <height> for each object in the image in new line. 

    Note:

    <object-class> integer object number from 0 to (classes-1) corresponding to bdd.names

    <x> <y> are float values relative to the width and height of the image and describe the center coordinate of the bounding box

    <w> <h> are float values relative to the width and height of the image
 
    The script also outputs a .txt file with the absolute paths to all of the images in the train and val sets
    The annotations file is stored in data/bdd/labels and the paths files are stored in data/bdd
    The image jpgs for the train and val sets are stored in data/bdd/images
    5.To train the patch based model on image crops, we created the scripts crop_with_pad.py and patch_train_file.py in data/bdd to generate 416 x 416 crops of the training set and generate the corresponding annotations file and image paths file. Some example crops can be found in data/bdd/example_crops
    6.Modify train.py and test.py to take the BDD dataset filepaths as input as well as any training parameters used

## Test
Evaluates the baseline model on BDD Dataset full size images. (our weights files are too large to store here)

    $ python3 test.py --weights_path checkpoints/uncropped.weights
    
To evaluate the patch-based model we created the following scripts to take in batches of crops, run detections on them, combine the crops into a full image, and calculate mAP:

    $ python3 modified_detect.py  

    $ python3 calculate_map.py detect_patchcrd.json data/bdd.names path/to/labels 


## Train
    train.py [-h] [--epochs EPOCHS] [--image_folder IMAGE_FOLDER]
                [--batch_size BATCH_SIZE]
                [--model_config_path MODEL_CONFIG_PATH]
                [--data_config_path DATA_CONFIG_PATH]
                [--weights_path WEIGHTS_PATH] [--class_path CLASS_PATH]
                [--conf_thres CONF_THRES] [--nms_thres NMS_THRES]
                [--n_cpu N_CPU] [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--checkpoint_dir CHECKPOINT_DIR]
```

## Credit
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
