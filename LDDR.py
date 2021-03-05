from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


dir_ind=0
right=0
left=0

def cannyfct(image):

 

    #Turning image gray applying gray filter so we can have pixel with intensities from 0 to 255dw
    Gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #Apllying GaussianBlur plus a kernel of 5 by 5 and a 0 deviation so we can reduce the noise
    blur=cv2.GaussianBlur(Gray,(5,5),0)
    #Applying canny method so we can determin edges by finding places where we got big differences between pixels intensities
    # 0 ---> 255 strong gradient and 0 ---> 20 small gradient
    # Widh = nr of collumns in the matrix and height = number of rows  in the matrix their product gives the resolution
    # I used a 1:1 ratio so i can detect both side lanes and central ones, we will probably go to 1:3 later
    canny=cv2.Canny(blur,50,150)
    
    
    return canny

def Interest_Polygon(image):
    
    
    height=image.shape[0]
    #Define the dimensions of the triangle
    polygons=np.array([
        [(0,height),(590,height),(450,275),(250,275)]
        ])
    #Creates an array of zeroes with the same dimension as canny
    mask = np.zeros_like(image)
    #Add the triangle over the mask
    cv2.fillPoly(mask,polygons,255)
    #Using bitwise and both images we restrict it to show us only the region represented by that polygon,the zone of interest
    Image_Mask=cv2.bitwise_and(image,mask)
    
    
    return Image_Mask

def Show_Lines(image,Lines):
    
 
    lane_lines=np.zeros_like(image)
    if Lines is not None:
        for line in Lines:
                global dir_ind,right,left
                X1,Y1,X2,Y2=line.reshape(4)
                cv2.line(lane_lines,(X1,X2),(Y1,Y2),(0,0,255),10) #besides coordinates we have color and the line thicknes
                cv2.rectangle(lane_lines,(X1,X2),(Y1,Y2),(0,255,0),3)
                #print(X1,"x1",X2,"x2",Y1,"y1",Y2,"y2",)
                if dir_ind == 0:
                    left=320-X1
                #    print("left:",left)
                    dir_ind=1
                else:
                    right=X1-320
                #    print("right:",right)
                    dir_ind=0

                #print("left:", left, "right:",right)
                #print(abs(320-X1),"Left")
                #print(abs(320-X2),"Right")
                #right=320-X2-rightzero
                #left=320-X1-leftzero
                #print(right,"right")
                #print(X1, "left")
                
                

                
    return lane_lines, right, left



def Direction_Calc(right, left):
    direction=left-right
    
    return direction


def make_coordinates(image,line_parameters):
    
     
        slope,intercept=line_parameters
        
         #Height width and number of chanels
        
        #Basic math 
        y1=image.shape[0]
        y2=int(y1*(3/4.3))
        x1=int((y1-intercept)/slope)
        x2=int((y2-intercept)/slope)
        
        return np.array([x1,x2,y1,y2])
   

def Slope_Intercept(image,Lines):
    #We will save the coordinates of the "average" lines on the left and right
    left_fit=[]
    right_fit=[]
    if Lines is not None:
        for line in Lines:
                X1,Y1,X2,Y2=line.reshape(4)
                
                parameters=np.polyfit((X1,X2) , (Y1,Y2),1)
                slope=parameters[0]
                intercept=parameters[1]
                if slope < 0:
                    left_fit.append((slope,intercept))
                else:
                    right_fit.append((slope,intercept))
        #print(left_fit)
        #print(right_fit)
        
    
   
    if len(right_fit)==len(left_fit)==0:
        return np.array([])
    elif len(right_fit)==0:
        left_fit_average=np.average(left_fit,axis=0)
        left_line=make_coordinates(image,left_fit_average)
        return np.array([left_line])
    elif len(left_fit)==0:
        right_fit_average=np.average(right_fit,axis=0)
        right_line=make_coordinates(image,right_fit_average)
        return np.array([right_line])
    
    
    right_fit_average=np.average(right_fit,axis=0)
    right_line=make_coordinates(image,right_fit_average)
    left_fit_average=np.average(left_fit,axis=0)
    left_line=make_coordinates(image,left_fit_average)
    return np.array([left_line,right_line])

#SOURCE IMAGES

#image=cv2.imread('test.png')

#Make a copy and NEVER work with array 
# lane_image=image is WRONG  since this way any modification on lane_image will also affect image

#lane_image=np.copy(image)

#Apllying filters



#Creating the mask




#Hough Lines used to detect the lane lines, wiht precision 2 pixels and 1 degree precision in radians and the minimal number of inteersctions in hough space
#The last 2 argguments represent the minimal length of the line which shall be considerated and the maximum gap between those intrerupted lines i n pixels of course


#Combining the images so we can see the lines detected on the original picture, gamma parameter with basic value


#PRINTING
#print(image.shape)


cap=cv2.VideoCapture("Test_Real.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    if frame is None:
        break
    frame=frame.astype("uint8")
    canny=cannyfct(frame)
    Crp_Img=Interest_Polygon(canny)
    Lines=cv2.HoughLinesP(Crp_Img,2,np.pi/180,100,np.array([]),minLineLength=2,maxLineGap=200)
    averaged_lines=Slope_Intercept(frame,Lines)
    lane_lines,right,left=Show_Lines(frame,averaged_lines)
    combo=cv2.addWeighted(frame,0.8,lane_lines,1,1)
    direction = Direction_Calc(right,left)
    print("Direction:", direction)
    time.sleep(0.1)
    
 #   center=cv2.line(combo,(320,300),(320,440),(0,0,255),10)
    cv2.imshow('Imaginea de test',combo )
   
    cv2.waitKey(1)
  
    #plt.imshow(Crp_Img)
    #plt.show()