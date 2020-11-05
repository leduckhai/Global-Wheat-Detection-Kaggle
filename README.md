# Global Wheat Detection (Kaggle) 

Tutorial YOLO v4 to detect wheat heads from crops. The project is based on this Kaggle Competition: https://www.kaggle.com/c/global-wheat-detection.

## Description 
In this competition, you’ll detect wheat heads from outdoor images of wheat plants, including wheat datasets from around the globe. Using worldwide data, you will focus on a generalized solution to estimate the number and size of wheat heads. To better gauge the performance for unseen genotypes, environments, and observational conditions, the training dataset covers multiple regions. You will use more than 3,000 images from Europe (France, UK, Switzerland) and North America (Canada). The test data includes about 1,000 images from Australia, Japan, and China.

## Examples
These examples have wheat heads detected

![alt text](https://user-images.githubusercontent.com/45472148/98257653-8a4eec80-1fb2-11eb-97d7-a01a0b07031f.png)

These examples don't have any wheat heads

![alt text](https://user-images.githubusercontent.com/45472148/98257663-8d49dd00-1fb2-11eb-86d5-2fc98e5546a3.png)

## Data-processing procedures
The data given in the contest have some problems:

**1. The given label format is not YOLO v4 format:**

The given is [x_min y_min width height], while YOLO v4's is [class x_center y_center width height]. 

![alt text](https://user-images.githubusercontent.com/45472148/98257687-9175fa80-1fb2-11eb-9042-c5216663c010.png)

So we have to convert into YOLO v4 format:

```
# Convert bounding box to YOLO v4
def convert_to_YOLOv4(list_of_annotation):
  x_min = list_of_annotation[0] 
  y_min = list_of_annotation[1] 
  width = list_of_annotation[2] 
  height = list_of_annotation[3] 

  width_img = 1024
  height_img = 1024

  # Conversion
  bx = round((x_min+width)/(2*width_img),6)
  by = round((y_min+height)/(2*height_img),6)
  w = round(width/width_img,6)
  h = round(height/height_img,6)

  annotation = [0,bx,by,w,h]
  
  return annotation
```

**2. Not all training data have labels:**

The number of images in "train.csv" file is less than the in "train" folder

![alt text](https://user-images.githubusercontent.com/45472148/98257700-933fbe00-1fb2-11eb-834d-085d97c28c17.PNG)

So we can only utilize images in "train.csv" and regretly waste the remnants

**3. There are only training and test data, missing validation data:**

So we have to split training and validation data in ratio 80/20

```
# Training/test data split
from sklearn.model_selection import train_test_split

image_id_train, image_id_test, bbox_train, bbox_test = train_test_split(image_id, bbox, test_size=0.2, random_state=42)
```

## Why use Darknet YOLO v4?
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

> Paper: https://arxiv.org/abs/2004.10934

> Source code: https://github.com/AlexeyAB/darknet

The result of YOLO v4 is impressive compared to other models:

![alt text](https://user-images.githubusercontent.com/45472148/98257679-8fac3700-1fb2-11eb-84a1-961fe21e7467.png)

