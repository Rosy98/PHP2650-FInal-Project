---
title: "PHP2650 Final Project"
author: "Yu Yan, Zihan Zhou"
date: "05/13/2023"
output:
  html_document:
    df_print: paged
link-citations: yes
editor_options:
  markdown:
    wrap: 72
bibliography: refs.bib
---

```{=html}
<style type="text/css">

h1.title {
  font-size: 38px;
  color: Black;
  text-align: center;
}
h4.author { /* Header 4 - and the author and data headers use this too  */
    font-size: 18px;
  font-family: "Times New Roman", Times, serif;
  color: Grey;
  text-align: center;
}
h4.date { /* Header 4 - and the author and data headers use this too  */
  font-size: 18px;
  font-family: "Times New Roman", Times, serif;
  color: Grey;
  text-align: center;
}
figure{text-align: center; max-width: 40%; margin:0;padding: 10px;}
figure img{width: 100%;}
</style>
```
# 1. Introduction

In the last three years or so, the need to diagnose and manage patients
has become more urgent than ever due to the outbreak of the world's
coronavirus disease 2019 (COVID-19). Chest X-rays (CXRs), one of the
most primary imaging tools, are common, fast, non-invasive, relatively
cheap and may be used to track the disease's development [@melba]. While
developing drugs to hinder virus prefoliation and new methods to assist
infected individuals, alongside making effective sanitary policies to
prevent virus spread are crucial endeavors of medical researchers, the
role of computer science, emphasized by its significant contributions
such as innovative technologies for virus diagnostics and tracking human
interactions, is equally vital in this fight against the virus
[@MARQUES2020106691].

Nowadays, scientists are employing Convolutional Neural Networks (CNN),
a class of deep learning neural networks for multiple applications. In
the 1860s, Wiesel and Hubel [@wiesel] studied the visual cortex cells of
cats and found that each visual neuron processes only a small area of
the visual image, the Receptive Field. And inputting the entire pixel
data to traditional Neural Network is highly inefficient and
computationally demanding. This inspired the concept of convolutional
neural networks (CNNs), a powerful and effective tool for image
classification because of its high accuracy. CNNs aim to automatically
learn relevant features from images by using an input layer, an output
layer, and hidden layers. Typically, the hidden layers comprise
convolutional layers, ReLU layers, pooling layers, and fully connected
layers. CNNs marks a significant breakthrough in automatic image
classification systems as that bypass the need for pre-processing of
images that wa a requirement in conventional machine learning algorithms
[@MARQUES2020106691].

The primary goal of our project is to develop a Convolutional Neural
Network (CNN)-based system for the classification of X-ray images. We
will provide a comprehensive explanation of what CNNs are and how they
operate within this context. During the training and testing phase, the
dataset has been divided into separate parts, which helps to validate
the proposed CNN models and helps prevent overfitting, a common issue in
machine learning models. The multi-class classification using images
from patients with COVID-19, pneumonia, and those who are healthy, are
discussed.

# 2. Dataset Description and Sources

The data we use is a clean dataset from kaggle website. These images are
collected from various publicly available resources:

-   COVID-19 image data collection [@melba]
    <https://github.com/ieee8023/covid-chestxray-dataset>

-   Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images
    for Classification [@Kermany2018b] [@KERMANY2018]
    <https://data.mendeley.com/datasets/rscbjbr9sj/2>

-   COVID-Net Open Source Initiative [@Wang2020]
    <https://github.com/lindawangg/COVID-Net>

The first source, is the initial publicly accessibel COVID-19 image
dataset, which is the biggest publicly available source for COVID-19
picture, offering a comprehensive collection of hundreds of frontal view
X-ray images [@melba]. The second dataset source collected and labels
chest X-ray images from children, which includes 3,883 instances of
pneumonia and 1,349 normal cases, taken from a total of 5,856 patients
[@KERMANY2018]. The third dataset source comes from COVID-Net open
source initiative, providing a collection of chest X-ray images for
different categories: no pneumonia, non-COVID-19 pneumonia, and COVID-19
pneumonia, taken from over 16,400 patients [@Wang2020].

The cleaned curated dataset from Kaggle, available at
<https://www.kaggle.com/code/faressayah/chest-x-ray-medical-diagnosis-with-cnn-densenet/input?select=Data>,
has arranged and split the above three dataset sources into two folders:
'train', 'test'. Each of these folders contains three subfolders,
representing three categories: 'COVID-19', 'PNEUMONIA', and 'NORMAL'.
The dataset includes a total of 6,432 X-ray images, with the test data
constituting 20% of the total images. The following are a few examples
from each category.

<p align="center">
<img src="images/X-rays/COVID19(108).jpg" width="20%"/>
<img src="images/X-rays/COVID19(463).jpg" width="20%"/>
<img src="images/X-rays/COVID19(501).jpg" width="20%"/>
<img src="images/X-rays/COVID19(539).jpg" width="20%"/> <br><br> Fig.1.COVID-19
</p>

<p align="center">
<img src="images/X-rays/PNEUMONIA(3443).jpg" width="20%"/>
<img src="images/X-rays/PNEUMONIA(3462).jpg" width="20%"/>
<img src="images/X-rays/PNEUMONIA(3614).jpg" width="20%"/>
<img src="images/X-rays/PNEUMONIA(3627).jpg" width="20%"/> <br><br> Fig.2.PNEUMONIA
</p>

<p align="center">
<img src="images/X-rays/NORMAL(1267).jpg" width="20%"/>
<img src="images/X-rays/NORMAL(1274).jpg" width="20%"/>
<img src="images/X-rays/NORMAL(1379).jpg" width="20%"/>
<img src="images/X-rays/NORMAL(1415).jpg" width="20%"/> <br><br> Fig.3.NORMAL
</p>


# 3. Understanding Convolutional Neural Networks

Convolutional Neural Network (CNN) is one kind of deep nural networks. The
capacity to classify images and identify objects in a picture has
improved significantly with the development of convolutional neural
networks [@DBLP2013]. Convolutional neural employs a special kind of
method which is being known as convolution. Suppose we have two
measurable functions on $\mathbb{R}^n$, $f$ and $g$, convolution is
defined as: 

$$(f*g)(t)=\int_{-\infty}^\infty f(\tau)g(t-\tau)d\tau$$

A main difference between traditional Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) lies in the dimensional structure of their layers. In CNNs, layers possess three dimensions - height, width, and depth, where 'depth' refers to the third dimension of an activation volume [@DBLP2015]. Consider the following fully connected layers, where each neuron in one layer connects to every neuron in the adjacent layer [@nielsen2015]. However, this design in ANNs does not take into account the spatial structure of images, treating input pixels that are both far apart and close together in an identical manner, which may hamper the network's ability to efficiently process image data [@nielsen2015].

<p align="center">
<img src="images/Model/full.png" width="50.0%"/> 
<br>
<br>
Fig.4.Fully-connected Layers [@nielsen2015]
</p>

In contrast, CNNs utilize convolution, as previously mentioned, to focus on local region of an image. Convolution is applied to a small region of an image, referred to as 'receptive field' or 'local region' instead of the entire image. As illustrated in Fig.6, to enhance efficiency, the hidden neurons in the next layer only get inputs from the corresponding part of the previous layer [@8308186]. This approach not only reduces computational requirements but also helps in recognizing spatial hierarchies within an image (Fig.5)[@8308186].

<p align="center">
<img src="images/Model/CNN.gif" width="30%"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="images/Model/convolution.gif" width="30%"/>
<br> 
<br>
Fig.5. Three dimensional input representation of CNN &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fig.6.Convolution as alternative for fully connected network
</p>

There are three distinct types of layers in CNNs: Convolutional, Pooling, and Fully-connected layers. Stacking these layers together forms a complete CNN architecture. As an example, Fig.7 depicts a simplified CNN architecture designed for MNIST digit classification[@DBLP2015].

<p align="center">
<img src="images/Model/architrcture.png" width="50.0%"/> 
<br>
<br>
Fig.7.An simple CNN architecture, comprised of just five layers [@DBLP2015]
</p>

Then we are 


# 4. Applictaions

## 4.1 Data Augmentation
Before we feed the train data into some model to train, we identified that we are lacking sufficient amount of image data.

To account for the insufficiency of data images, we included the structure of image data generator. It is basically a form of data augmentation Technic in the context of image data. We created artificial data that is consisted of transformations of original image without acquiring new images. This would increase the size of train data set to make the model training process more robust. In the <b>keras</b> package, the function <b>image_data_genertor()</b> completes such task. We could specify the transformations that we would like to achieve. For example, we can flip the images horizontally and vertically, changing contrast and hue of the images, zooming, shearing and also changing brightness of the images. Here we are implementing the following changes:

- rescale by a factor of 1/255
- shift width and height by a factor of 0.2
- shear and zoom the image by a factor of 0.2
- make both horizontal and vertical flip 
- make whitening and set brightness range to be 0.2

## 4.2 Model Structure
we have build this sequential model using tidy format under <b>keras</b> in R. As denoted above, the model is a combination of three component: input layer, hidden layers (consisted of convolution layer and pooling layer), and an output layer. All the convolution layers are using the activation function of 'relu'.

The input layer is a convolution layer with 32 filters and a 3,3 kernel size. The input_shape should be specified and matched what we set beforehand. We have 64,64,3 since we expect the input image to be at dimension of 64 by 64 and 3 means we are expecting RGB option. If we have grey scale set, we would instead insert 1 at this block. Remember to append a pooling layer after each convolution layer to wrap up the feature information extracted by the convolution layer filtering. 

Then we have three convolution layers with filter numbers respectively 64,128,and 128. This is simply the result of our exploration and training, users can have their own exploration over the layers and number of filters to train the model. By inserting a flatten layer, we are end with the convolution part and moved on to the typical networks to perform classification assignment.

Start with a drop out layer of 0.5, we added two dense layer with units 128 and 64. This would convert information of image features to make classification task. 

The final layer, output layer is a dense layer with three units, since we have three labels. We set the activation function to be softmax so that the model will finally give its prediction of probabilities for each label of any given image. And the three probabilities should sum up to 1.

Now that we are finally able to compile the model with our constructed train and test dataset from the last section. We want the loss function to be 'categorical_crossentropy', and optimization algorithm to be Adam with a learning rate of 0.0001. <b>TensorFlow</b> enables a great deal of flexibilities here that user can try out different optimization algorithm and learning rate. And we also want the model to output accuracy so that we could evaluate.

Let's fit the mode with train data and evaluate on test data! The epochs are set to be 30 which means the training process will go through the entire train data 30 times. To accelerate training time, we added the option of multiprocessing and an early stopping criteria by patience being 5 in terms of accuracy check, so that the model train will stop earlier if detected convergence.

# 5. Discussion

## 5.1 Results

## 5.2 Conclusion

# 6. Future Work

To be added.

# Reference
