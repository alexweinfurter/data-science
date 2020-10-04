---
title: "Dog breed classification"
date: 2020-08-27
excerpt: "Classifying images of dogs according to their breed with Convolutional Neural Networks (CNN)"
header:
   image: /images/dog-breed-classification/schafer-dog.jpg
   thumbnail: /images/dog-breed-classification/schafer-dog.jpg
---

## Motivation
Since dogs have been domesticated by humans they evolved from a useful working animal to the humans best friend. 
Nowadays you'll see them in every park where humans are going for a walk with them. But don't you wonder sometimes to which breed a dog belongs? This problem could be solved by using an image classification application to predict the dog breed.

## Objectives
As part of my Data Science Nanodegree I implement such a dog breed classifier. The classifier should consist of three different steps:

1. The classifier must be able to detect and distinguish between humans and dogs in images.
2. If the image contains the face of a human the breed that most resembles the human in appearance should be returned.
2. If the image contains a dog, it's breed should be returned.


## Methodology
Using images of humans and dogs and leveraging face detection algorithms as well as Convolutional Neural Networks and Transfer Learning the objectives stated above should be achievable. The following steps are performed:
1. Preprocessing of the images to use them with OpenCV and Keras.
2. Using a <a href="https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html">Haar-feature based CascadeClassifier</a> proposed by Paul Viola and Michael Jones to detect human faces. 
3. Using a pretrained Resnet50 classifier to detect dogs in an image.
4. Training of a Convolutional Neural Network to predict the dog breed.
5. Training of a better Convolutional Neural Network by leveraging Transfer Learning using the bottleneck features of a pretrained Resnet50 model.


## Dataset
Before going into the classification details, we have a look at the two datasets used in this project. One dataset consists of images of humans (13233 in total), the other one contains dog images (8351 in total).
There are 133 different dog breeds in the dog dataset which are distributed as shown in the barplot below.
![alt]({{ site.url }}{{ site.baseurl }}/images/dog-breed-classification/breed_distribution.png)


The dog dataset has been split into independet training, validation and test datasets with the following sizes:
* Training: 6680 images
* Validation: 835 images
* Test: 836 images


## Human face detection
In order to detect human beings in images a pretrained <a href="https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html">Haar-feature based CascadeClassifier</a> from the OpenCV framework is used. OpenCV provides a bunch of different models which are saved as XML files and can be downloaded <a href="https://github.com/opencv/opencv/tree/master/data">here on GitHub</a>. The model can be loaded by simply instantiating a CascadeClassifier:
```python
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
```

To use the classifier the images are converted to grayscale. Afterwards the classifier can be used easily by passing the grayscaled image. 
```python
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

```
As you can see in the image below it even outputs the location (x/y-coordinates and width + height of a rectangle) of the detected face.
![alt]({{ site.url }}{{ site.baseurl }}/images/dog-breed-classification/human.png)
In this project it's only necessary to know if a face exists in an image at all. Therefore the length of the 'faces' array is checked to decide whether there is a human or not:
```python
len(faces) > 0
```

## Dog detection
To detect if an image contains a dog a pretrained ResNet-50 model, a popular CNN architecture, is used. The ResNet-50 model has been trained on the ImageNet dataset and thus with more than 10 million images belonging to one of 1000 categories. In total 117 of these categories represent different dog breeds which is why the pretrained model should be able to detect dogs without difficulties.

In this project Keras with a TensorFlow backend has been used to implement the CNNs. Keras provides functionality to load a pretrained ResNet-50 model:
```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```
Before using this model it is important to correctly preprocess the given images. The model expects to use a 4-dimensional tensor in the following shape:

  (#samples, 224, 224, 3).

Here <code>#samples</code> is the number of images used, 224 is the width and height of the images and 3 corresponds to the RGB values of the pixels.
With Keras it's easy to get a tensor of this shape, as the following code shows.
```python
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type and resize it
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

Furthermore it is necessary to reorder the color channels from RGB to BGR and to center the each color channel with respect to the ImageNet dataset the model has been trained with. This can be achieved by using a preprocessing function provided by Keras:
```python
from keras.applications.resnet50 import preprocess_input
img = preprocess_input(path_to_tensor(img_path))
```

When using the <code>predict</code> function of the model, it's important to know that it returns probabilities for each possible class. Therefore the index of the class with the maximum probability is the class label of interest. The following function has been implemented to predict the labels of given image.
```python
from keras.applications.resnet50 import preprocess_input

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```
A dictionary containing the corresponding human readable class labels can be found <a href="https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a">here</a>.


## Future work
To enhance the implemented classification model the following steps should be considered:
1. Easy idea but not that easy to do: get a larger training dataset
2. Augment the given image data by using the corresponding Keras functionalities. The images could e.g. be rotated, shifted, zoomed, etc..
3. Deal with the class imbalance (some breeds have more than twice as many instances as others) .e.g. by using <code>class_weights</code>. This would influence the loss function and give higher weights to rare classes.
