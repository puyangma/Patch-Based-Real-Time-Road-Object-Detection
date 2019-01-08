import sys
import os
from PIL import Image, ImageOps
import glob

INPUT_IMAGE_WIDTH = 720
INPUT_IMAGE_HEIGHT = 1280

# This function adds padding to the right and bottom of the images if necessary to make them square, 
# and then scales them to a default size of 416 x 416.
def process_image(img, file_name, desired_size=416):
    assert (img.size[0] in (720, 1280)), print("ERROR: Image " + file_name + " has unexpected width of " + str(img.size[0]))
    assert (img.size[1] in (720, 1280)), print("ERROR: Image " + file_name + " has unexpected height of " + str(img.size[1]))

    # Add padding if necessary.
    if img.size[0] > img.size[1]: # If width is larger.
        delta_height = img.size[0] - img.size[1]
        padding = (0, 0, 0, delta_height)
        img = ImageOps.expand(img, padding)
    elif img.size[0] < img.size[1]: # If height is larger.
        delta_width = img.size[1] - img.size[0] 
        padding = (0, 0, delta_width, 0)
        img = ImageOps.expand(img, padding)

    img = img.resize((desired_size, desired_size), Image.ANTIALIAS)
    return img

def main(argv):
    input_directory = argv[1]
    output_directory = argv[2]

    files = sorted(glob.glob('%s/*.*' % input_directory)) 
    for file in files:
        img = Image.open(file).copy()
        file_name = os.path.basename(file)
        img = process_image(img, file_name) 
        img.save(output_directory + "/" + file_name)           

# Example usage: python3 process_images.py data/images/10k/train data/images/10k/train_processed
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("ERROR: Invalid input arguments.")
    else: 
        main(sys.argv)
