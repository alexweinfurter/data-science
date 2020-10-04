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
As part of my Data Science Nanodegree I implement such a dog breed classifier. The classifier consists of three different steps:

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

In this project it's only necessary to know if a face exists in an image at all. Therefore the length of the <code>faces</code> array is checked to decide whether there is a human or not:
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
A dictionary containing the corresponding human readable class labels can be found <a href="https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a">here</a>. If the predicted class label belongs to a dog breed we know that there is a dog in the image.

## Dog breed classification from Scratch
To classify the dog breeds at first a Convolutional Neural Network has been trained from scratch. Before defining the architecture of the CNN it is necessary to rescale the images so that the RGB values are in the interval [0,1]:
```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

Afterwards an uncomplex CNN architecture has been defined:
```python
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(2,2), strides=(1, 1), padding='same', input_shape=train_tensors.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None))

model.add(Conv2D(filters=32, kernel_size=(2,2), strides=(1, 1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None))

model.add(Conv2D(filters=64, kernel_size=(2,2), strides=(1, 1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None))

#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(266, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(133, activation='softmax'))
```
Convolutional layers have been used for feature extraction and max-pooling to downsample the data and reduce the dimensionality. Initially a GlobalAveragePooling layer has been used to drastically reduce the dimensionality of the feature map. A Dense layer with 133 nodes (corresponding to the class labels) has been used for the predictions. This model only achieved around 1.5% accuracy.

Therefore I've decided to flatten the feature map from the convolutions+maxpooling instead of using the GlobalAveragePooling. The features are afterwards used by two fully connected Dense layers (and a Dropout layer in between to avoid overfitting). This results in around 5.6% accuracy which is still poor but much better than the 1.5% achieved with the architecture proposed above. There is definitly a lot of room for improvement, as you will see in the next section.

## Transfer Learning for dog breed classification
Transfer Learning means that an already trained neural network is used to perform a new, previously unknown, task. This can reduce the training time while increasing the model accuracy.

To leverage Transfer Learning the weights of a pretrained model are loaded. Then there are two possibilities to fine-tune the model:
1. Perform further training one the whole pretrained neural network.
2. Perform further training only on the output layer of the neural network.

In this project the second approach has been used. The feature extraction capabilities of pretrained Convolutional Neural Networks have been used to extract so called *bottleneck features*. These abstract features are then used for further training of a Dense layer to learn the classification task.

For the Transfer Learning task two CNN architectures have been considered: the VGG16 and the ResNet-50. To obtain the bottleneck features the models have been loaded and used to predict the features. The parameter <code>include_top</code> has been set to false to ensure only the features are extracted and no classification is performed.
```python
def extract_VGG16(tensor):
  from keras.applications.vgg16 import VGG16, preprocess_input
  return VGG16(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Resnet50(tensor):
  from keras.applications.resnet50 import ResNet50, preprocess_input
  return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
```

On top of the extracted bottleneck features a GlobalAveragePooling layer has been used to reduce the tensor dimensionality and prevent overfitting (alternatively e.g. using Flatten()+Dense() layers is likely to lead to overfitting, for more information see <a href="https://arxiv.org/pdf/1312.4400.pdf">this paper</a>, sections 3.2 and 4.6). The resulting feature map is used by a Dense layer to learn the classification task at hand and output the probability of each class. 

```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

resnet_model = Sequential()
resnet_model.add(GlobalAveragePooling2D(input_shape=train_resnet50.shape[1:]))
resnet_model.add(Dense(133, activation='softmax'))

VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
resnet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

These models have achieved much better results than the CNN built from scratch. The VGG-based model achieves 38.3% accuracy and the ResNet-50-based model an accuracy of 81.2%. Both models have been trained much faster than self-made CNN.

## Put it all together
At the end the face and dog detector have been combined with the best breed classifier, the ResNet-50-based model in this case. 
The algorithm gets an image path as argument and first decides whether a prediction is made for a human or a dog and then outputs the most resembling breed.

```python
def breed_detector(img_path):
    if face_detector(img_path):
        print("human detected")
        return Resnet50_predict_breed(img_path)
    elif dog_detector(img_path):
        print("dog detected")
        return Resnet50_predict_breed(img_path)
    else:
        print("Neither face nor dog found in image.")
        return None
```

In the images below you can see some of the predictions made by the <code>breed_detector</code>.


![alt]({{ site.url }}{{ site.baseurl }}/images/dog-breed-classification/pred_schafer_dog.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/dog-breed-classification/pred_johnny_depp.png)


![alt]({{ site.url }}{{ site.baseurl }}/images/dog-breed-classification/pred_saint_bernard.png)

## Future work
To enhance the implemented classification model the following steps could be considered:
1. Easy idea but not that easy to do: get a larger training dataset.
2. Augment the given image data by using the corresponding Keras functionalities. The images could e.g. be rotated, shifted, zoomed, etc..
3. Deal with the class imbalance (some breeds have more than twice as many instances as others), e.g. by using <code>class_weights</code>. This would influence the loss function and give higher weights to rare classes.
4. Using ensemble or stacking methods to combine the predictions of several different models.
