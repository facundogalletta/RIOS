import math
import numpy as np

FOV=78.8 #° Diagonal

w=4000
h=2250
d=np.sqrt(w**2+h**2)

FOV=math.radians(FOV)

FOVw=2*np.arctan(w/(d/np.tan(FOV/2)))
FOVh=2*np.arctan(h/(d/np.tan(FOV/2)))

FOVw=math.degrees(FOVw)
FOVh=math.degrees(FOVh)

print(FOVw,FOVh)


