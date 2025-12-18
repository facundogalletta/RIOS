import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from find_vertices_new import find_vertices_new

width=5280
height=2970
FOVwidth=64.686982896768800
FOVheight=39.211354832645280
band=0
img_type=3


x=[100,100,100,100]
y=[100,100,100,100]
pitch= [20,20,20,20]
roll = [5,5,5,5]
yaw = [0,10,20,30]
h=[50,50,50,50]

xlim=[np.min(x)-200,np.max(x)+200]
ylim=[np.min(y)-200,np.max(y)+200]
zlim=[0,np.max(h)+10]

for i in range(len(pitch)):
    [x1_n, y1_n], [x2_n, y2_n], [x3_n, y3_n], [x4_n, y4_n] = find_vertices_new(x[i], y[i], h[i], np.deg2rad(yaw[i]), np.deg2rad(pitch[i]), np.deg2rad(roll[i]), FOVwidth,FOVheight)    
    [x1_0, y1_0], [x2_0, y2_0], [x3_0, y3_0], [x4_0, y4_0] = find_vertices_new(x[i],y[i],h[i] , 0, 0, 0, FOVwidth,  FOVheight)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot([x1_n, x2_n], [y1_n, y2_n], [0, 0], color='g',label='Region oblique')
    ax.plot([x2_n, x4_n], [y2_n, y4_n], [0, 0], color='g')
    ax.plot([x4_n, x3_n], [y4_n, y3_n], [0, 0], color='g')
    ax.plot([x3_n, x1_n], [y3_n, y1_n], [0, 0], color='g')
    
    ax.scatter([x1_n], [y1_n], [0], color='darkblue',label='x1')
    ax.scatter([x2_n], [y2_n], [0], color='skyblue',label='x2')
    ax.scatter([x3_n], [y3_n], [0], color='yellow',label='x3')
    ax.scatter([x4_n], [y4_n], [0], color='g',label='x4')

    
    origin = np.array([x[i], y[i], h[i]])
    ax.plot([origin[0],x1_n], [origin[1],y1_n], [origin[2], 0],color='r',linestyle='--', label='FOV oblique')
    ax.plot([origin[0],x2_n], [origin[1],y2_n], [origin[2], 0],color='g',linestyle='--',)
    ax.plot([origin[0],x3_n], [origin[1],y3_n], [origin[2], 0],color='g',linestyle='--',)
    ax.plot([origin[0],x4_n], [origin[1],y4_n], [origin[2], 0],color='g',linestyle='--',)
    
    ax.plot([x1_0, x2_0], [y1_0, y2_0], [0, 0], color='r',label='Region zenithal')
    ax.plot([x2_0, x4_0], [y2_0, y4_0], [0, 0], color='r')
    ax.plot([x4_0, x3_0], [y4_0, y3_0], [0, 0], color='r')
    ax.plot([x3_0, x1_0], [y3_0, y1_0], [0, 0], color='r')

    ax.plot([origin[0],x1_0], [origin[1],y1_0], [origin[2], 0],color='r',linestyle='--', label='FOV zenithal')
    ax.plot([origin[0],x2_0], [origin[1],y2_0], [origin[2], 0],color='r',linestyle='--',)
    ax.plot([origin[0],x3_0], [origin[1],y3_0], [origin[2], 0],color='r',linestyle='--',)
    ax.plot([origin[0],x4_0], [origin[1],y4_0], [origin[2], 0],color='r',linestyle='--',)
    
    font = {'fontname':'Times New Roman','fontweight':'bold','fontsize':14}
    ax.set_xlabel('X',**font)
    ax.set_ylabel('Y',**font)
    ax.set_zlabel('Z',**font)

    # Invertir el eje Z
    ax.invert_zaxis()
    # Agregar una leyenda
    ax.legend()
    #invertir eje Z
    ax.invert_zaxis()
    
    #Set limites de ejes
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

plt.show()


    
    
