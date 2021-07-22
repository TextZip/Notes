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
array_name = np.arange(start_value,end_value,step_value)
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
 array_name = np.zeros(shape,data_type)
```
An array with all "ones" can be created using 
```python
 array_name = np.ones(shape,data_type)
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

### Numpy and Images 

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