import json
import cv2
import os.path
from PIL import Image, ImageOps
import glob

# define parameters
injson='bdd100k/labels/bdd100k_labels_images_val.json'
folder='images/val2014'
outfolder='images/val2014_cropped'
winsize=416
stride=416

f= open(injson)
data = json.load(f)
L = 2000#len(data)


for i in range(0,L):

    if os.path.isfile(folder+'/'+data[i]['name']) :
        print(data[i]['name']+' is being processed')
        img=cv2.imread(folder+'/'+data[i]['name'])
        a=data[i]['labels']

        for n in range(0,img.shape[0],stride):
            for m in range(0,img.shape[1],stride):
              c=data[i].copy()
              c['labels']=[]
              w = winsize if (n+winsize<=img.shape[0]) else (img.shape[0]-n)
              h = winsize if (m+winsize<=img.shape[1]) else (img.shape[1]-m)
              pImg=img[n:n+w,m:m+h,:]
              c['name']=data[i]['name'][0:-4]+'_'+str(n)+'_'+str(m)+'.jpg'

              if pImg.shape[1] > 40: #eliminate tiny crops with no useable information

                  cv2.imwrite(outfolder+'/'+c['name'],pImg)



#Add padding for rectangle crops
files = sorted(glob.glob('%s/*.*' % outfolder))
for file in files:
  pImg = Image.open(file).copy()
  img_name = os.path.basename(file)
  #add padding if rectangle
  current_w = pImg.size[0]

  current_h = pImg.size[1]
  deltaw = (winsize - current_w)
  deltah = (winsize - current_h)
  if winsize>pImg.size[0] and winsize>pImg.size[1]:
    padding = (0,0,deltaw,deltah)
    pImg = ImageOps.expand(pImg,padding)

  if winsize>pImg.size[0]:
    padding = (0,0,deltaw,0)
    pImg = ImageOps.expand(pImg,padding)

  if winsize>pImg.size[1]:
    padding = (0,0,0,deltah)
    pImg = ImageOps.expand(pImg,padding)
  pImg.save(outfolder + "/" + img_name)
