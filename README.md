## Part 2

## Mountain Finding

### Problem Statement

In this part, we need to identify the shape of mountains (ridgeline) in the test images provided. The detected ridgeline needs to be highlighted with three different colors. 
Legend:
1. Red - Detection using Bayes Net
2. Blue - Detection using Viterbi Algorithm
3. Green - Detection using Viterbi Algorithm + Human Input of a coordinates of point on ridgeline


### Preprovided Code and Data :

1. mountain.py - code that reads in an image file and produces an **edge strength map** that measures how strong the image gradient (local contrast) is at each point.
2. test_images - Images of mountains on which we need to detect the ridgeline

### Approach to the Problem

#### 1. Bayes Net
With the Bayes Net implementation, we are searching for the highest pixel gradient in each column of the image. The **edge strength** function provided in the code returns a 2D array in which each element gives the strength of image gradient at that pixel in the image.
Using the argmax function of numpy on edge_strength 2D array would return the indexes of high strength image gradient in each column of the image. We used these indexes to highlight the ridgeline on the mountain image.

#### Pros and Cons:
1. Mountain Images in which the ridgeline has the strongest gradient in the image are easily tracked by this method. One such example is below:
							[![output-5.jpg](https://i.postimg.cc/bvGGVrJg/output-5.jpg)](https://postimg.cc/gwWknY3L)
 
 2. Images in which the ridgeline is not the strongest gradient in the image are difficult to track with this method, because it will track other features in the image which have high pixel gradient. One such example is below:
		 [![output-88.jpg](https://i.postimg.cc/1zcxQcf3/output-88.jpg)](https://postimg.cc/673bVZrk)
 
#### 2. Viterbi

The other method for ridge-line detection is use of Viterbi Algorithm. The key feature of implementing viterbi is that once it detects the ridge-line in one of the column iterations, it would track the rest of the ridgeline.
We start by finding the emission probablities of all the pixels in column 0 of the image, by dividing the edge strength at each pixel by the total edge strength of all pixels in the column.
Next, we defined the transition probabalities, which would basically define the probabilities of finding a ridge-line pixel along the rows. We took into consideration that the next column pixel which is adjacent to the current one will have max probability of being a ridge-line, but since the mountains go up and down as well, we tweaked and decreased the transition probablity of transitions to upper or lowers rows from current state. We tested various transitional probabilities, and decided to go with below one :

    for i in range(0, col_pixels):  
    trans_p[i] = 1 - i * 100 / col_pixels

 We propogate columnwise and check max probability value by multiplying emission probabilities and transition probabilities, we keep updating max probability value whenever a state is encountered which has a high probability value compared to current max value. We also store the row index from previous column which gave that max probability value which will be used later to track back the max-indexes in each column i.e. the ridgeline.
 After each row pixel iteration, we update the image state array by updating it's value by taking into consideration the max probability for each pixel, so we multiply the pixel values in edge_strength array with the corresponding max value of that index.   

### 2.1 Backpropogation
Once we reach the end of the image by updating our state probabilities and updating the max probabilities for each pixel in the image. We again use the argmax function on the last column of state prob array to get the index of row which has highest pixel gradient value. 
Also, we have stored the index of row from previous column which maxmized this pixel probability, so we track back the entire image back and store these indices in an array.
We will use this array to superimpose on the original image resulting in the ridge-line.

### Pros and Cons:

1. Since in our program we start propogation from 0th column, we expect the ridge-line pixel to have the highest probability, so the tracking can start from that pixel. This will result in accurate tracking of the ridge-line. One such example is below:
		[![mt8-vit-test-1.jpg](https://i.postimg.cc/m2zr7Ddy/mt8-vit-test-1.jpg)](https://postimg.cc/6yKwX9x8)

2. In the case, where in the 0th column the max initial probability of the pixel is not that of the ridge-line pixel, then the viterbi algorithm loses its path and may or may not catch up with the ridge-line. One such example is below:
		[![mt6-vit-test-2.jpg](https://i.postimg.cc/LXWpskMK/mt6-vit-test-2.jpg)](https://postimg.cc/DJ1DjXZB)
		
	The initial pixel gradient selected was not of the ridge-line so viterbi algorithm was not able to track down the ridge-line.

### 3. Viterbi + Human Input

One way we could eliminate the above issue in viterbi method of missing the ridge-line, would be to introduce a human input of a coordinate of a point on the ridge-line.
Then, from that point we would do backward and forward propogation of the viterbi algorithm.

We would use the same updated image_state array and max_tracker array from our viterbi method, to get more accurate results.

#### Important Point - We need to maximize the point(pixel) which is being given as human input, or else during the backtracking, the algorithm would skip that pixel and ridge-line would get messed up, as in the below test image.
[![mt1-human-test-1.jpg](https://i.postimg.cc/nVTdqVsr/mt1-human-test-1.jpg)](https://postimg.cc/7bGMkknr)

We did this by making the whole column 0, and only maximizing that one row index as 1.

### Coordinates used for Human Input for each Image:

1. (74, 77)
2. (56, 152)
3. (43, 160)
4. (55, 141)
5. (59, 93)
6. (72, 95)
7. (20, 83) and (52 24)
8. (64, 125)
9. (70, 169)

### Some Final Images -

[![Final-2.jpg](https://i.postimg.cc/rFB5r79z/Final-2.jpg)](https://postimg.cc/tYz10SwH)

[![Final-9.jpg](https://i.postimg.cc/QMb7vCZW/Final-9.jpg)](https://postimg.cc/7C5b2x6x)

### Challenges Faced - 

1. Finding the optimized transitional probabilities was difficult, as some values were working for one image and not for another.
2. Maximizing the human input was key in getting correct results. David explained the same a few times in the office hours, and it helped!
3. We were getting a numpy overflow error when we were updating the image_state array. We fixed it by dividing a large number (1000) to get the result in the limits.




