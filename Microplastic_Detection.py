# 1. Input image resizing
import cv2 as cv
import matplotlib.pyplot as plt

img = plt.imread("input_image.png")  # input image

scale_percent = 38.5  # output image size percentage
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# output image
resized_img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

#------------------------------------------------------------------------------------------------------------------------------------------------------

# 2. Gaussian derivative calculation to obtain image gradients
import numpy as np
import math

def DoGx(w, sigma):  # "w" is the matrix size while "sigma" controls the width of the decay rate of the Gaussian Filter
    maxX=w//2  # maximum right side distance
    minX=-maxX # maximum left side distance
    minY=minX  # maximum bottom distance
    maxY=maxX  # maximum top distance
    DoGx_arr=np.zeros((w,w))  # X Derivative of Gaussian array (DoGx)
    for x in range(minX, maxX+1): # x iteration from minX to maxX
        for y in range(minY, maxY+1): # y iteration from minY to maxY
            res = (-x/(2*math.pi*(sigma**4))) * math.exp(-(x**2+y**2)/(2*(sigma**2))) # res = DoGx equation
            DoGx_arr[y-minY, x-minX]=res # fill DoGx array
    return DoGx_arr

def DoGy(w, sigma):
    maxX=w//2   # maximum right side distance
    minX=-maxX # maximum left side distance
    minY=minX  # maximum bottom distance
    maxY=maxX  # maximum top distance
    DoGy_arr=np.zeros((w,w)) # Y Derivative of Gaussian array (DoGY)
    for x in range(minX, maxX+1): # x iteration from minX to maxX
        for y in range(minY, maxY+1): # y iteration from minY to maxY
            res = (-y/(2*math.pi*(sigma**4))) * math.exp(-(x**2+y**2)/(2*(sigma**2))) # res = DoGy equation
            DoGy_arr[y-minY, x-minX]=res # fill DoGy array
    return DoGy_arr

DOGx=DoGx(3,0.5)    # call DoGx()
DOGy=DoGy(3,0.5)    # call DoGy()

# Obtaining image Gradients
img_SC=np.float64(resized_img[:,:,0]) # Convert input image into 2-Dimensional array (float64)
Gx=cv.filter2D(img_SC,-1,DOGx)
Gy=cv.filter2D(img_SC,-1,DOGy)

# Gradient vectors
Gm=(Gx**2 + Gy**2)**0.5
phi=np.rad2deg(np.arctan2(Gy,Gx))

#------------------------------------------------------------------------------------------------------------------------------------------------------

# 3. Non-Maxima Suppression (NMS)
def NMS(Gm, phi):
    phi_bin = 45 * (np.round(phi / 45))  # gradient direction quantification
    Gm_NMS = np.zeros(Gm.shape)  # Gm array
    rows, cols = Gm.shape[0], Gm.shape[1]
    p, r = 0., 0.  # 2 adjacent pixels
    for row in range(1,
                     rows - 1):
        for col in range(1, cols - 1):
            angle = phi_bin[row, col]
            if angle == 180. or angle == -180. or angle == 0. or angle == -0.:
                p, r = Gm[row, col + 1], Gm[row, col - 1]
            elif angle == 135. or angle == -135.:
                p, r = Gm[row + 1, col + 1], Gm[row - 1, col - 1]
            elif angle == 90. or angle == -90.:
                p, r = Gm[row - 1, col], Gm[row + 1, col]
            elif angle == 45. or angle == -45.:
                p, r = Gm[row - 1, col + 1], Gm[row + 1, col - 1]
            else:
                print("error")
                return

            if Gm[row, col] >= p and Gm[row, col] >= r:  # if q>=p and q>=r
                Gm_NMS[row, col] = Gm[row, col]  # set q and r intensities as 0

    return Gm_NMS


Gm_NMS = NMS(Gm, phi)  # NMS output

#------------------------------------------------------------------------------------------------------------------------------------------------------

# 4. Thresholding
Th=0.035 # threshold value (chosen based on the best output)
im_edge=Gm_NMS>Th # thresholding result (edge image)

#------------------------------------------------------------------------------------------------------------------------------------------------------

# 5. Excessive edge pixel elimination
idx=np.argwhere(im_edge) # edge pixel coordinates (True values)
idx_lst=[] # edge pixel coordinate list
for i in range(idx.shape[0]): # iteration i from 0 to (edge coordinate-1)
    idx_lst.append([])
    idx_lst[i]=idx[i] # convert edge pixel coordinate array into the edge pixel coordinate list

index = 1
diff = 10 # maximum difference between 2 adjacent pixels
while index < len(idx_lst):  # eliminate elements with absolute difference <=10
    if abs(idx_lst[index][0] - idx_lst[index - 1][0]) <= diff and abs(idx_lst[index][1] - idx_lst[index - 1][1]) <= diff:
        del idx_lst[index]
    else:
        index += 1

idx_new=np.array(idx_lst) # convert list back into the array

#------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. Create voting array "H"
def Hough_Circles(im_edge, idx_new):
    rows, cols = im_edge.shape[0], im_edge.shape[1]  # a=rows, b=cols
    rho_max = int((rows ** 2 + cols ** 2) ** 0.5)  # maximum circle radius
    H = np.zeros((rows, cols, rho_max))  # Hough voting array "H"
    r, c = idx_new[:, 0], idx_new[:, 1]  # r and c edge pixel coordinates

    for i in range(len(r)):
        for a in range(rows):
            for b in range(cols):
                ri, ci = r[i], c[i]
                rho_i = int(((ri - a) ** 2 + (ci - b) ** 2) ** 0.5)  # rho=((r-a)^2 + (c-b)^2)^0.5
                if rho_i > 0 and rho_i < rho_max:
                    H[a, b, rho_i] += 1  # add vote to the circle (a,b,rho_i)
    return H

H = Hough_Circles(im_edge, idx_new)

#------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Pick n-elements of H with the largest votes
uniques=np.unique(H) # sort H elements by values (smallest to biggest)
uniques_new=uniques[::-1] # re-sorted H elements (biggest to smallest)

n=7 # initial number prediction of microplastic particles

y_lst,x_lst,rad_lst=[],[],[] # define 3 different lists to be filled y, x, and rho
for i in range(n): # iteration of i from 0 to n:
    y,x,rad=np.where(H==uniques_new[i]) # y=a, x=b, rad=rho
    for id in range(len(y)):
        y_lst.append(y[id])
        x_lst.append(x[id])
        rad_lst.append(rad[id])

y_lst2, x_lst2, rad_lst2 = [y_lst[0]], [x_lst[0]], [rad_lst[0]]  # create new coordinate lists

ind = 1
while ind < len(y_lst):  # while iteration (ind) < (y_list length-1)
    y_min = list(range(y_lst[ind] - 5, y_lst[ind] + 1))  # create a list to store values from (y_lst[ind]-5) to (y_lst[ind])
    y_plus = list(range(y_lst[ind], y_lst[ind] + 6))   # create a list to store values from (y_lst[ind]) to (y_lst[ind]+5)
    x_min = list(range(x_lst[ind] - 5, x_lst[ind] + 1))  # create a list to store values from (x_lst[ind]-5) to (x_lst[ind])
    x_plus = list(range(x_lst[ind], x_lst[ind] + 6))  # create a list to store values from (x_lst[ind]) to (x_lst[ind]+5)

    if any([any(ele in y_lst2 for ele in y_min), any(ele in y_lst2 for ele in y_plus)]) and any(
            [any(ele in x_lst2 for ele in x_min), any(ele in x_lst2 for ele in x_plus)]):
        ind += 1


    else:
        y_lst2.append(y_lst[ind])
        x_lst2.append(x_lst[ind])
        rad_lst2.append(rad_lst[ind])
        ind += 1

#------------------------------------------------------------------------------------------------------------------------------------------------------

# 8. Plot the remaining circles
im_edge2=np.zeros((im_edge.shape[0],im_edge.shape[1],3), dtype=np.uint8) # create a three-dimensional array (RGB image)
im_edge3=np.array(255*im_edge,dtype=np.uint8)   # create an edge image consisting of values 0 and 255
im_edge2[:,:,0]=im_edge3   # insert im_edge3 to each channel of im_edge2
im_edge2[:,:,1]=im_edge3
im_edge2[:,:,2]=im_edge3

for i in range (len(y_lst2)):
    cv.circle(im_edge2,(int(x_lst2[i]),int(y_lst2[i])),int(rad_lst2[i]),(0,0,255),1)
    # visualize each resulting circle with the color red and width = 1

plt.imshow(im_edge2[:,:,::-1]) # plot the resulting image
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------

# 9. Print the number of circle(s) (detected microplastics)
print("Number of microplastic particle(s):",len(y_lst2))
