## Introduction
The following notes on computer vision is based on OpenCV and python.

## Numpy
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
A typical image from the internet can be represented by a matrix of the shape `(height, width, no_of_channels)` where the number of channels is usually 3 for color and 1 for grayscale.

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

## Image Basics 
**Reading Images using CV2**
```python
import cv2
img = cv2.imread('DATA/00-puppy.jpg')
while True:
    cv2.imshow('Puppy',img)
    # EXPLANATION FOR THIS LINE OF CODE:
    # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163
    # IF we've waited at least 1 ms AND we've pressed the Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
```

**Reading Grayscale version of Images using CV2**
```python
>>> img_gray = cv2.imread('../DATA/00-puppy.jpg',cv2.IMREAD_GRAYSCALE)
>>> plt.imshow(img_gray,cmap='gray')
```
***NOTE:** CV2 uses BGR color scheme where as matplotlib uses RGB, to convert from one color space to another use.*
```python
>>> img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
>>> plt.imshow(img_rgb)
```

**Resizing Images (Absolute)**
```python
>>> new_array =cv2.resize(image_array,(new_width, new_height))
```
**Resizing Images (Ratio)**
```python
>>> new_img =cv2.resize(img_rgb,(0,0),img,w_ratio,h_ratio)
```

**Flipping Images**
```python
# Along central x axis
>>> new_img = cv2.flip(new_img,0)
>>> plt.imshow(new_img)
```
![test](/assets/img/Computer-Vision/puppy_flip1.png)

```python
# Along central y axis
>>> new_img = cv2.flip(new_img,1)
>>> plt.imshow(new_img)
```
![test](/assets/img/Computer-Vision/puppy_flip2.png)

```python
# Along both axis
>>> new_img = cv2.flip(new_img,-1)
>>> plt.imshow(new_img)
```
![test](/assets/img/Computer-Vision/puppy_flip3.png)

**Saving Image Files**
```python
>>> cv2.imwrite('my_new_picture.jpg',new_img)
```

### Drawing on Images
**Creating a black canvas**
```python
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import cv2

>>> blank_img = np.zeros(shape=(512,512,3),dtype=np.int16)
>>> plt.imshow(blank_img)
```
![test](/assets/img/Computer-Vision/canvas.png)



**Drawing Rectangles**
```python
# pt1 = top left
# pt2 = bottom right
>>> cv2.rectangle(blank_img,pt1=(384,0),pt2=(510,128),color=(0,255,0)thickness=5)
>>> plt.imshow(blank_img)
```
![test](/assets/img/Computer-Vision/rect1.png)

```python
# pt1 = top left
# pt2 = bottom right
>>> cv2.rectangle(blank_img,pt1=(200,200),pt2=(300,300),color=(0,0,255),thickness=5)
>>> plt.imshow(blank_img)
```
![test](/assets/img/Computer-Vision/rect2.png)



**Drawing Circles**
```python
>>> cv2.circle(img=blank_img, center=(100,100), radius=50, color=(255,0,0), thickness=5)
>>> plt.imshow(blank_img)
```
![test](/assets/img/Computer-Vision/circle1.png)
**Drawing Solid shapes**
```python
>>> cv2.circle(img=blank_img, center=(400,400), radius=50, color=(255,0,0), thickness=-1)
>>> plt.imshow(blank_img)
```
![test](/assets/img/Computer-Vision/circle2.png)



**Drawing Lines**
```python
# Draw a diagonal blue line with thickness of 5 px
>>> cv2.line(blank_img,pt1=(0,0),pt2=(511,511),color=(102, 255, 255),thickness=5)
>>> plt.imshow(blank_img)
```
![test](/assets/img/Computer-Vision/line.png)



**Drawing Text**
```python
>>> font = cv2.FONT_HERSHEY_SIMPLEX
>>> cv2.putText(blank_img,text='Hello',org=(10,500), fontFace=font,fontScale= 4,color=(255,255,255),thickness=2,lineType=cv2.LINE_AA)
>>> plt.imshow(blank_img)
```
![test](/assets/img/Computer-Vision/text.png)



**Drawing Polygons**
```python
>>> blank_img = np.zeros(shape=(512,512,3),dtype=np.int32)
>>> vertices = np.array([[100,300],[200,200],[400,300],[200,400]],np.int32)
>>> vertices = vertices.reshape((-1,1,2))
>>> cv2.polylines(blank_img,[vertices],isClosed=True,color=(255,0,0),thickness=5)
>>> plt.imshow(blank_img)
```
![test](/assets/img/Computer-Vision/polygon.png)

**Drawing with mouse**
```python
import cv2
import numpy as np


# Create a function based on a CV2 Event (Left button click)
drawing = False # True if mouse is pressed
ix,iy = -1,-1

# mouse callback function
def draw_rectangle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        # When you click DOWN with left mouse button drawing is set to True
        drawing = True
        # Then we take note of where that mouse was located
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        # Now the mouse is moving
        if drawing == True:
            # If drawing is True, it means you've already clicked on the left mouse button
            # We draw a rectangle from the previous position to the x,y where the mouse is
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
           

    elif event == cv2.EVENT_LBUTTONUP:
        # Once you lift the mouse button, drawing is False
        drawing = False
        # we complete the rectangle.
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        
        

# Create a black image
img = np.zeros((512,512,3), np.uint8)
# This names the window so we can reference it 
cv2.namedWindow(winname='my_drawing')
# Connects the mouse button to our callback function
cv2.setMouseCallback('my_drawing',draw_rectangle)

while True: #Runs forever until we break with Esc key on keyboard
    # Shows the image window
    cv2.imshow('my_drawing',img)
    # EXPLANATION FOR THIS LINE OF CODE:
    # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163
    
    # CHECK TO SEE IF ESC WAS PRESSED ON KEYBOARD
    if cv2.waitKey(1) & 0xFF == 27:
        break
# Once script is done, its usually good practice to call this line
# It closes all windows (just in case you have multiple windows called)
cv2.destroyAllWindows()
```

## Image Processing
**RGB Color Space**
![test](/assets/img/Computer-Vision/RGB.png)



**HSV Color Space**
![test](/assets/img/Computer-Vision/HSV.png)



**HSL Color Space**
![test](/assets/img/Computer-Vision/HSL.png)


### Color Mapping
**BGR to RGB**
```python
>>> img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
>>> plt.imshow(img)
```
![test](/assets/img/Computer-Vision/puppy_numpy.png)

**BGR to HSV**
```python
>>> img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
>>> plt.imshow(img)
```
![test](/assets/img/Computer-Vision/BGR2HSV.png)

**BGR to HSL**
```python
>>> img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
>>> plt.imshow(img)
```
![test](/assets/img/Computer-Vision/BGR2HSL.png)



### Blending, Pasting & Masks

### Thresholding

### Blurring & Smoothing 

<!-- ## Video Basics

## Object Detection

## Object Tracking 

## Deep Learning -->