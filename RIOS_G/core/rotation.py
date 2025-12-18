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

pitch= np.deg2rad(20)
roll= np.deg2rad(5)
yaw= np.deg2rad(0)
h=50

############################################################################
# DETERMINAR VECTORES QUE APUNTAN EN DIRECCIÓN DE LAS ESQUINAS DE LA IMAGEN#
############################################################################

FOVw = np.deg2rad(FOVwidth)  # Ángulo de visión de la cámara en la dirección larga de la foto
FOVh = np.deg2rad(FOVheight)  # Ángulo de visión de la cámara en la dirección corta de la foto

y1= h * np.tan(FOVh/2)
x1=-h * np.tan(FOVw / 2)

y2= y1
x2= -x1

y3=-y1
x3=x1

y4=-y1
x4=-x1

v1=np.array([x1,y1,-h])
v2=np.array([x2,y2,-h])
v3=np.array([x3,y3,-h])
v4=np.array([x4,y4,-h])

######################
# MATRIZ DE ROTACIÓN #
######################

CY=np.cos(-yaw)
SY=np.sin(-yaw)
CP=np.cos(pitch)
SP=np.sin(pitch)
CR=np.cos(roll)
SR=np.sin(roll)

M1=np.array([[CY,SY,0], [-SY,CY,0], [0,0,1]])
M2 = np.array([[1, 0, 0], [0,CP,-SP], [0, SP, CP]])
M3 = np.array([[CR, 0, SR], [0, 1, 0], [-SR, 0, CR]])
M=M1.dot(M2.dot(M3))


####################
# VECTORES ROTADOS #
####################

v1_p=np.dot(M,v1)
v2_p=np.dot(M,v2)
v3_p=np.dot(M,v3)
v4_p=np.dot(M,v4)

[x1_0,y1_0], [x2_0,y2_0], [x3_0,y3_0], [x4_0,y4_0] =find_vertices_new(150, 150, h, 0, 0, 0, FOVw,  FOVh)

print([y1_0, x1_0], [y2_0, x2_0], [y3_0, x3_0], [y4_0, x4_0])

[x1_n, y1_n], [x2_n, y2_n], [x3_n, y3_n], [x4_n, y4_n] =find_vertices_new(150, 150, h, yaw, pitch, roll, FOVw,FOVh)

print([y1_n, x1_n], [y2_n, x2_n], [y3_n, x3_n], [y4_n, x4_n])


# Crear una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotear los vectores
origin = np.array([150, 150, h])

ax.quiver(*origin, *v1*0.25, color='k', label='$\\hat{v_i}$',linewidth = 2)
ax.quiver(*origin, *v2*0.25, color='k',linewidth = 2)
ax.quiver(*origin, *v3*0.25, color='k',linewidth = 2)
ax.quiver(*origin, *v4*0.25, color='k',linewidth = 2)

ax.quiver(*origin, *v1_p*0.25, color='b', label='$\\hat{v_i}^,$',linewidth = 2)
ax.quiver(*origin, *v2_p*0.25, color='b',linewidth = 2)
ax.quiver(*origin, *v3_p*0.25, color='b',linewidth = 2)
ax.quiver(*origin, *v4_p*0.25, color='b',linewidth = 2)

ax.plot([x1_n, x2_n], [y1_n, y2_n], [0, 0], color='g')
ax.plot([x2_n, x4_n], [y2_n, y4_n], [0, 0], color='g')
ax.plot([x4_n, x3_n], [y4_n, y3_n], [0, 0], color='g')
ax.plot([x3_n, x1_n], [y3_n, y1_n], [0, 0], color='g')

ax.plot([origin[0],x1_n], [origin[1],y1_n], [origin[2], 0],color='g',linestyle='dotted',label='$R = ($'+str(round(np.rad2deg(yaw)))+'$^o,$'+str(round(np.rad2deg(pitch)))+'$^o,$'+str(round(np.rad2deg(roll)))+'$^o)$')
ax.plot([origin[0],x2_n], [origin[1],y2_n], [origin[2], 0],color='g',linestyle='dotted',)
ax.plot([origin[0],x3_n], [origin[1],y3_n], [origin[2], 0],color='g',linestyle='dotted',)
ax.plot([origin[0],x4_n], [origin[1],y4_n], [origin[2], 0],color='g',linestyle='dotted',)


ax.plot([x1_0, x2_0], [y1_0, y2_0], [0, 0], color='r')
ax.plot([x2_0, x4_0], [y2_0, y4_0], [0, 0], color='r')
ax.plot([x4_0, x3_0], [y4_0, y3_0], [0, 0], color='r')
ax.plot([x3_0, x1_0], [y3_0, y1_0], [0, 0], color='r')

ax.plot([origin[0],x1_0], [origin[1],y1_0], [origin[2], 0],color='r',linestyle='--',label='$R_0 = (0^o,0^o,0^o)$')
ax.plot([origin[0],x2_0], [origin[1],y2_0], [origin[2], 0],color='r',linestyle='--',)
ax.plot([origin[0],x3_0], [origin[1],y3_0], [origin[2], 0],color='r',linestyle='--',)
ax.plot([origin[0],x4_0], [origin[1],y4_0], [origin[2], 0],color='r',linestyle='--',)

ax.scatter([150],[150],[h], color='y',label='Camera',linewidth=3)


font = {'fontname':'Times New Roman','fontweight':'bold','fontsize':14}

ax.set_xlabel('X',**font)
ax.set_ylabel('Y',**font)
ax.set_zlabel('Z',**font)

# Invertir el eje Z
ax.invert_zaxis()

# Apagar el grid de los planos verticales
# ax.zaxis._axinfo['grid'].update(color = 'white', linestyle = '--', linewidth = 0)
# ax.yaxis._axinfo['grid'].update(color = 'w', linestyle = '-', linewidth = 0)
# ax.xaxis._axinfo['grid'].update(color = 'w', linestyle = '-', linewidth = 0)

ax.w_zaxis.set_pane_color((.8, 0.9, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

# Agregar una leyenda
ax.legend()
#invertir eje Z
ax.invert_zaxis()

# Mostrar el gráfico
plt.show()



    




