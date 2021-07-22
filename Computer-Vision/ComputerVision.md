## Introduction
The following notes on computer vision is based on OpenCV and python.

## Numpy and Image Basics
We will be using numpy for doing low-level math operations on images, the images are represented as matrices.

### Arrays
**Creating numpy arrays**
```python
>>> array_name = np.array([1,2,3,4])
```
Some properties of the above created array:
```python
>>> array_name
array([1, 2, 3, 4])

>>> type(array_name)
numpy.ndarray
```
**Uniformly distributed arrays in numpy**

```python
>>> array_name = np.arange(start_value,end_value,step_value)
```
*Note: The `start_value` is inclusive and the `end_value` is exclusive.*

The above command is similar to range in python, some properties and outputs of the above command:

```python
>>> even = np.arrange(0,20,2)
>>> even
array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
>>> even.shape
(10,)
>>> even.reshape(5,2)
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10],
       [12, 14],
       [16, 18]])
```
The above set of commands can be simplified into a single line
```python
>>> even = np.arange(0,20,2).reshape(5,2)
>>> even
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10],
       [12, 14],
       [16, 18]])
```
`object_name.shape` can be used to display the current shape of an numpy object.
`np.reshape()` can be used to change the shape of the array.

**Converting python lists to numpy arrays**
```python
>>> my_list = [1,2,3,4]
>>> numpy_array = np.array(my_list)
>>> numpy_array
array([1, 2, 3, 4])
>>> type(numpy_array)
numpy.ndarray
```

**Creating standard numpy arrays**
An array with all "zeros" can be created using 
```python
>>> array_name = np.zeros(shape,data_type)
```
An array with all "ones" can be created using 
```python
>>> array_name = np.ones(shape,data_type)
```
Some examples of the above functions
```python
>>> array1 = np.zeros((5,5),np.uint8)
>>> array1
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]], dtype=uint8)
>>> array2 = np.ones((3,2),np.int32)
>>> array2
array([[1, 1],
       [1, 1],
       [1, 1]], dtype=int32)
```
**Indexing and Slicing**
Numpy arrays can be indexed and sliced like regular python lists.
Some examples of the same:
```python
### Creating a sample array ###
>>> big_array = np.arange(0,100).reshape(10,10)
>>> big_array
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
       [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
       [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
       [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
       [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])

### Indexing ###
>>> row = 6
>>> column = 9
>>> big_array[row,column]
69

### Slicing ###
>>> column_five = big_array[:,5]
>>> column_five
array([ 5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
>>> row_six = big_array[6,:]
>>> row_six
array([60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
>>> small_array = big_array[5:,4:8]
>>> small_array
array([[54, 55, 56, 57],
       [64, 65, 66, 67],
       [74, 75, 76, 77],
       [84, 85, 86, 87],
       [94, 95, 96, 97]])
```

### Numpy and Images 
A tipical image from the internet can be represented by a matrix of the shape `(height, width, no_of_channels)` where the number of channels is usually 3 for color and 1 for grayscale.

**Opening Images using PIL**
```python
>>> from PIL import Image

>>> image_array = Image.open('path/to/image.png')

>>> type(image_array)
PIL.JpegImagePlugin.JpegImageFile
```

**Converting Images into Numpy-Arrays**
```python
>>> from PIL import Image
>>> import numpy as np

>>> image_array = Image.open('path/to/image.png')
>>> image_np = np.asarray(image_array)

>>> type(image_np)
numpy.ndarray
```

**Displaying Images**
```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from PIL import Image

>>> puppy_image = Image.open('00-puppy.jpg')
>>> type(puppy_image)
<class 'PIL.JpegImagePlugin.JpegImageFile'>
>>> puppy_image
<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1950x1300 at 0x7F82AF887BE0>
```
![test](/assets/img/Computer-Vision/puppy_original.png)
```python
>>> puppy_array = np.asarray(puppy_image)
>>> type(puppy_array)
<class 'numpy.ndarray'>
>>> puppy_array.shape
(1300, 1950, 3)
>>> plt.imshow(puppy_array)
<matplotlib.image.AxesImage object at 0x7f82af7b9490>
```
![test](/assets/img/Computer-Vision/puppy_numpy.png)

**Image Channels**

## Image Basics 

### Drawing on Images

## Image Processing

### Color Mapping

### Blending, Pasting & Masks

### Thresholding

### Blurring & Smoothing 

## Video Basics

## Object Detection

## Object Tracking 

## Deep Learning