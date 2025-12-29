import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib_scalebar.scalebar import ScaleBar  # Librería externa

# --- Raster ---
Path="D:/GeoMapperDron/PruebaTermica/1/Geo_Term_D_A_NS_55_225/"

raster_path = Path + "Mosaic_6.tif"
csv_path = Path + "temp.csv"
kml_path = "D:/fgalletta/OneDrive - Facultad de Ingeniería/01_Investigación/04_Publicaciones/Paper Dron/03-Avances/Termica/Figuras/Sensores/Sensores Punta del Tigre.kml"

with rasterio.open(raster_path) as src:
    dst_crs = "EPSG:3857"
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    dst_array = np.empty((height, width), dtype=src.dtypes[0])
    reproject(
        source=rasterio.band(src, 1),
        destination=dst_array,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )

# --- KML puntos ---
gdf = gpd.read_file(kml_path, driver='KML')
gdf = gdf.to_crs(dst_crs)

# --- Raster con máscara ---
min_temp = 15
masked_array = np.where(dst_array < min_temp, np.nan, dst_array)

# --- Crear figura con gridspec ---
fig = plt.figure(figsize=(15, 15))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.2], hspace=0.05)

# --- Mapa arriba ---
ax_map = fig.add_subplot(gs[0])

# Basemap primero

# Raster encima (ajustamos un poco a la izquierda: 5% del ancho)
dx = (transform.a * width) * 0.05
im = ax_map.imshow(masked_array, extent=(transform.c - dx, transform.c + transform.a*width - dx,
                                         transform.f + transform.e*height, transform.f),
                   origin='upper', cmap='jet', vmin=21, vmax=27, zorder=2)

# Scatter puntos

x_center=-6295000
y_center=-4130000

half_side = 1000  # metros

xmin = -6296000+2220
xmax = -6294000+950
ymin = -4131000-630
ymax = -4129000-2050

ax_map.set_xlim(xmin, xmax)
ax_map.set_ylim(ymin, ymax)


gdf.plot(ax=ax_map, edgecolor='k', marker='o', color='y', markersize=40, alpha=0.8, zorder=3)

# Colorbar a la izquierda
divider = make_axes_locatable(ax_map)
cax = divider.append_axes("left", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Temperature (°C)', fontsize=14, labelpad=15)

ctx.add_basemap(ax_map, crs=dst_crs, source=ctx.providers.Esri.WorldImagery, zorder=1)

# Barra de escala métrica
scalebar = ScaleBar(dx=1, units="m", dimension="si-length", location='lower right',
                    pad=0.5, border_pad=0.5, color='white', box_color='black', box_alpha=0.5,
                    scale_loc='bottom')
ax_map.add_artist(scalebar)

# Quitar ticks del eje
ax_map.set_xticks([])
ax_map.set_yticks([])

# --- Gráfica abajo ---
ax_plot = fig.add_subplot(gs[1])
df = pd.read_csv(csv_path)
temp_media = df['temp_media']
temp_5 = df['temp_5']
temp_95 = df['temp_95']

ax_plot.plot(temp_media)
ax_plot.plot(temp_5, linestyle='dashed')
ax_plot.plot(temp_95, linestyle='dashed')
ax_plot.grid()
ax_plot.set_xlabel('Image', fontsize=18)
ax_plot.set_ylabel('Temperature (°C)', fontsize=18)
ax_plot.tick_params(axis='both', labelsize=14)
# ax_plot.legend(['Mean','Percentile 5%','Percentile 95%'], fontsize=14)

# --- Guardar figura completa ---
plt.savefig(Path + "Mapa_y_Temp.png", dpi=300, bbox_inches='tight')
plt.show()



# # --- Crear solo figura map


# Path="D:/GeoMapperDron/PruebaTermica/1/Geo_Term_RAW_2_120/"

# # Scatter puntos

# x_center=-6295000
# y_center=-4130000

# half_side = 1000  # metros

# xmin = -6296000+1800
# xmax = -6294000+1200
# ymin = -4131000-600
# ymax = -4129000-1200

# for i in range(2,120):
#     plt.figure(figsize=(15, 12))
#     ax_map = plt.axes()
#     with rasterio.open(Path + f"IMG_{i:04d}_6_georeferenced.tif") as src:
#         dst_crs = "EPSG:3857"
#         transform, width, height = calculate_default_transform(
#             src.crs, dst_crs, src.width, src.height, *src.bounds
#         )
#         dst_array = np.empty((height, width), dtype=src.dtypes[0])
#         reproject(
#             source=rasterio.band(src, 1),
#             destination=dst_array,
#             src_transform=src.transform,
#             src_crs=src.crs,
#             dst_transform=transform,
#             dst_crs=dst_crs,
#             resampling=Resampling.nearest
#         )
#     min_temp = 15
#     masked_array = np.where(dst_array < min_temp, np.nan, dst_array)

#     # Raster encima (ajustamos un poco a la izquierda: 5% del ancho)
#     dx = (transform.a * width) * 0.05
#     im = ax_map.imshow(masked_array, extent=(transform.c - dx, transform.c + transform.a*width - dx,
#                                             transform.f + transform.e*height, transform.f),
#                     origin='upper', cmap='jet', vmin=18, vmax=30, zorder=2)


#     ax_map.set_xlim(xmin, xmax)
#     ax_map.set_ylim(ymin, ymax)


#     gdf.plot(ax=ax_map, edgecolor='k', marker='o', color='y', markersize=40, alpha=0.8, zorder=3)

#     # Colorbar a la izquierda
#     divider = make_axes_locatable(ax_map)
#     cax = divider.append_axes("left", size="5%", pad=0.05)
#     cbar = plt.colorbar(im, cax=cax, orientation='vertical')
#     cbar.ax.yaxis.set_ticks_position('left')
#     cbar.ax.yaxis.set_label_position('left')
#     cbar.ax.tick_params(labelsize=14)
#     cbar.set_label('Temperature (°C)', fontsize=14, labelpad=15)

#     ctx.add_basemap(ax_map, crs=dst_crs, source=ctx.providers.Esri.WorldImagery, zorder=1)

#     # Barra de escala métrica
#     scalebar = ScaleBar(dx=1, units="m", dimension="si-length", location='lower left',
#                         pad=0.5, border_pad=0.5, color='white', box_color='black', box_alpha=0.5,
#                         scale_loc='bottom')
#     ax_map.add_artist(scalebar)

#     # Quitar ticks del eje
#     ax_map.set_xticks([])
#     ax_map.set_yticks([])

#     plt.savefig(f"D:/fgalletta\OneDrive - Facultad de Ingeniería/01_Investigación/04_Publicaciones/Paper Dron/03-Avances/Termica/Figuras/Vuelo/IMG_{i:04d}.png", dpi=300, bbox_inches='tight')
# plt.show()

