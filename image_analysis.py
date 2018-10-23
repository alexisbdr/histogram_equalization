import cv2 as cv
import numpy as np
import itertools
import matplotlib.pyplot as plt

def hist(img):
    
    hist_bins = np.zeros(256)
    y_i, x_i = img.shape
    #Create image histogram 
    for y, x in itertools.product(range(y_i), range(x_i)):
        hist_bins[img[y,x]] += 1 
    
    return hist_bins / (y_i*x_i)

def hist_equil(img):
   
    y_i, x_i = img.shape

    norm_hist = hist(img)
    
    #Normalize the cum sum histogram 
    normsum_hist = np.cumsum(norm_hist)
    
    #Transfer functions
    transf_func = np.uint8(255*normsum_hist)
    
    #Apply values
    new_img = np.zeros_like(img)
    for a,b in itertools.product(range(y_i), range(x_i)):
        new_img[b,a] = transf_func[img[b,a]]
    
    new_hist = hist(new_img) 
    
    return new_hist, new_img, norm_hist
    

def main():
    
    #Read in the image
    path = "moon.bmp"
    img = cv.imread(path, 0)
    cv.imshow(path , img)
    hist_equilized, img_equil , hist = hist_equil(img)
    cv.imshow("equil", img_equil)
    cv.imwrite("new_moon.bmp",img_equil)
    plt.plot(hist_equilized)
    plt.show()
    key = cv.waitKey(0)
    
    if key == 27 or key == 'q': 
        cv.destroyAllWindows() 


if __name__ == "__main__" :
    main()
