import numpy as np
from osgeo import ogr
from ArmadoMosAuxiliares import LongLat2UTM
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime

num_shp=15
num_id=3


W=np.zeros((num_shp,num_id))
X=np.zeros((num_shp,num_id,2))
Y=np.zeros((num_shp,num_id,2))

for k in range(15):
    
    shapefile_path = '../QGis_Shapes/b'+str(k+1)
    file = ogr.Open(f'{shapefile_path}.shp')
    x=np.zeros((1,num_id))
    y=np.zeros((1,num_id))
    shape = file.GetLayer(0)
    
    for i in range(num_id):
        feature = shape.GetFeature(i)
        geom=feature.GetGeometryRef()
        
        long_0=geom.GetX(0)
        lat_0=geom.GetY(0)
        long_1=geom.GetX(1)
        lat_1=geom.GetY(1)
    
        [x_UTM_0,y_UTM_0]=LongLat2UTM(long_0,lat_0,2)
        [x_UTM_1,y_UTM_1]=LongLat2UTM(long_1,lat_1,2)
    
        X[k,i,0]=x_UTM_0
        X[k,i,1]=y_UTM_1
        Y[k,i,0]=y_UTM_0
        Y[k,i,1]=y_UTM_1
        
        #distancia entre puntos
        W[k,i]=np.sqrt((x_UTM_1-x_UTM_0)**2+(y_UTM_1-y_UTM_0)**2)

#Media de anchos por fila
W_mean=np.mean(W,axis=1)

ind=len(W_mean)-3

W_mean=W_mean[0:len(W_mean)-3]
#Ordenar en cada fila
W_sort_fila=np.sort(W,axis=1)
W_sort_fila=W_sort_fila[0:ind,:]

print(W_mean)
print(W_sort_fila)

#Leer archivo de exel con horas
file=pd.read_excel('../QGis_Shapes/Horas.xlsx',header=None)
t0=file[0][0].hour*3600+file[0][0].minute*60+file[0][0].second
t=[]
for i in range(len(file[0])):
    t.append(file[0][i].hour*3600+file[0][i].minute*60+file[0][i].second-t0)
t=t[0:ind]


print(t)
print(W_mean)

index_max=np.argmax(W_mean)
c = np.polyfit(t[0:index_max+1], W_mean[0:index_max+1]**2/32, 1)
f = np.poly1d(c)

# index_max_0=np.argmax(W[:,0])
# c_1=np.polyfit(t[0:index_max_0+1],W_sort_fila[0:index_max_0+1,0]**2/32,1)
# f_1=np.poly1d(c_1)

# index_max_1=np.argmax(W[:,1])
# c_2=np.polyfit(t[0:index_max_1+1],W_sort_fila[0:index_max_1+1,1]**2/32,1)
# f_2=np.poly1d(c_2)

# index_max_2=np.argmax(W[:,2])
# c_3=np.polyfit(t[0:index_max_2+1],W_sort_fila[0:index_max_2+1,2]**2/32,1)
# f_3=np.poly1d(c_3)


b_0=4
t_0=-5
D=1.8

plt.figure()
plt.scatter(t,W_mean,s=30,marker='s',color='r',label='Media')
plt.errorbar(t,W_mean,yerr=W_sort_fila[:,2]-W_mean,lolims=np.ones(ind),uplims=np.zeros(ind),linestyle='none',color='k',capsize=3)

t_a=np.linspace(0,120)
print(t_a)
pol=b_0+np.sqrt(32*D*(t_a-t_0))

plt.plot(t_a,pol,color='b',label=r'$b = b_0+\sqrt{32 \cdot D \cdot (t+t_0)}$,  $b_0$= $4$ $m$, $D$= $1.8$ $m/s^2$, $t_0$= $5$ $s$' ,linestyle='-')

plt.errorbar(t,W_mean,yerr=W_mean-W_sort_fila[:,0],uplims=np.ones(ind),lolims=np.zeros(ind),linestyle='none',color='k',capsize=2,label='Desviaciones')
font = {'fontname':'Times New Roman','fontweight':'bold','fontsize':18}
plt.grid()
plt.xlabel('Tiempo (s)',**font)
plt.ylabel('b (m)',**font)
plt.legend()
plt.show()


# y_mean_ajuste=W_mean[0:index_max+1]**2/32
# y_ajuste_0=W_sort_fila[0:index_max+1,0]**2/32
# y_ajuste_1=W_sort_fila[0:index_max+1,1]**2/32
# y_ajuste_2=W_sort_fila[0:index_max+1,2]**2/32
# t_ajuste=t[0:index_max+1]

# plt.figure()
# plt.scatter(t_ajuste,y_mean_ajuste,s=20,marker='s',color='r',label='Media')
# plt.errorbar(t_ajuste,y_mean_ajuste,yerr=y_ajuste_2-y_mean_ajuste,lolims=np.ones(len(y_mean_ajuste)),uplims=np.zeros(len(y_mean_ajuste)),linestyle='none',color='k',capsize=3)
# plt.errorbar(t_ajuste,y_mean_ajuste,yerr=y_mean_ajuste-y_ajuste_0,uplims=np.ones(len(y_mean_ajuste)),lolims=np.zeros(len(y_mean_ajuste)),linestyle='none',color='k',capsize=3,label='Desviaciones')
# plt.plot(t_ajuste,f(t_ajuste),'--b',label='Pendiente ' + str(np.round(c[0],1)) + r' $m^2$/s')

# font = {'fontname':'Times New Roman','fontweight':'bold','fontsize':14}
# plt.grid()
# plt.xlabel(r'$Tiempo (s)$',**font)
# plt.ylabel(r'$W_y$ $^2$ / $32$ $(m^2)$ ',**font)
# plt.legend()
# plt.show()

