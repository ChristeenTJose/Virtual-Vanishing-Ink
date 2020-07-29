import cv2
import numpy as np
from caliberation import Caliberation

VANISHING_SPEED = 0.06 # Default : 0.06

THRESHOLD_SAT = 40 # Default : 40

THRESHOLD_VAL = 150 # Default : 150


if __name__ == "__main__":
	c=input("Press Enter to Caliberate marker: ")	
	Lower,Upper=Caliberation()
	c=input("\n\nPress Enter to start................... ")	
	vc = cv2.VideoCapture(0)
	x0,y0=-1,-1
	color=(0,255,255)#Default : Yellow
	temp=np.full(shape=(480,640,3),fill_value=(0,0,0),dtype=np.uint8)
	while cv2.waitKey(1)==-1:
		temp2=np.full(shape=(480,640,3),fill_value=(0,0,0),dtype=np.uint8)
		return_value,frame = vc.read()
		frame=cv2.flip(frame,1)
		hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		mask=cv2.inRange(hsv,np.array(Lower),np.array(Upper))
		kernel=np.ones((5,5),np.uint8)
		erosion=cv2.erode(mask,kernel,iterations=1)
		frame=np.array(frame)
		contours,hierarchy=cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		if contours:	
			c = max(contours, key = cv2.contourArea)
			x,y,W,H = cv2.boundingRect(c)
			if x0==-1:
				x0,y0=x+W//2,y+H//2
			else:
				cv2.line(temp,(x0,y0),(x+W//2,y+H//2),color,5)
				cv2.line(temp2,(x0,y0),(x+W//2,y+H//2),color,5)
				x0,y0=x+W//2,y+H//2
		else:
			x0,y0=-1,-1	
		
		
		
		# First mask : Transparent Region of Ink - using temp
		
		mask=cv2.cvtColor(temp,cv2.COLOR_BGR2HSV)
		mask=cv2.inRange(mask,np.array([1,THRESHOLD_SAT,THRESHOLD_VAL]),np.array([255,255,255]))# Set threshold to avoid digital stain
		
		temp=cv2.addWeighted(frame,VANISHING_SPEED,temp,1-VANISHING_SPEED,0) # Comment this line to get virtual ink
		
		
		temp=cv2.bitwise_and(temp,temp,mask=mask)
		mask=cv2.bitwise_not(mask)
		frame=cv2.bitwise_and(frame,frame,mask=mask)
		frame=cv2.add(frame,temp)
		
		# Second mask : Opaque Region of Ink - using temp2
		mask=cv2.cvtColor(temp2,cv2.COLOR_BGR2HSV)
		mask=cv2.inRange(mask,np.array([1,1,1]),np.array([255,255,255]))
		temp2=cv2.bitwise_and(temp2,temp2,mask=mask)
		mask=cv2.bitwise_not(mask)
		frame=cv2.bitwise_and(frame,frame,mask=mask)
		frame=cv2.add(frame,temp2)
		
		cv2.imshow('Frame',cv2.resize(frame,(1280,720)))		
	vc.release()
	cv2.destroyAllWindows()

	


