# Morphological Image Processing to Remove Noise

## Description

Given a binary image with noise, Morphological Operations are used to remove this noise. In this case, an image of a number of coins is given, where these coins are not distinct due to noise. Using morphological operations, these coins are separated out and counted.


## Data

The noisy binary image considered here is as below:

![alt text](/data/image.png)


## Approach

* Read image.

* Image erosion to separate the boundaries of the coins.

* Image dilatin to enlarge the boundaries.

* Detect number of coins using OpenCV Blob Detection.


## Output

![alt text](/output/output1.jpg)

![alt text](/output/output2.png)


## Getting Started

### Dependencies

<p align="left"> 
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>&ensp; </a>
<a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://www.codebykelvin.com/learning/python/data-science/numpy-series/cover-numpy.png" alt="numpy" width="40" height="40"/>&ensp; </a>
<a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://avatars.githubusercontent.com/u/5009934?v=4&s=400" alt="opencv" width="40" height="40"/>&ensp; </a>

* [Python 3](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [OpenCV](https://opencv.org/)


### Executing program

* Clone the repository into any folder of your choice.
```
git clone https://github.com/ninadharish/Morphological-Image-Processing-to-Remove-Noise.git
```

* Open the repository and navigate to the `src` folder.
```
cd Morphological-Image-Processing-to-Remove-Noise/src
```

* Run the program.
```
python main.py
```


## Authors

👤 **Ninad Harishchandrakar**

* [GitHub](https://github.com/ninadharish)
* [Email](mailto:ninad.harish@gmail.com)
* [LinkedIn](https://linkedin.com/in/ninadharish)
