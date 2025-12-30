
<p align="center">
  <img src="./docs/rios_logo.svg" alt="RIOS logo" width="600">
</p>

---

**RIOS** (Remote sensing with drone Imagery for Observation of water Surfaces) is a Python-based system designed for the processing and analysis of images of water bodies acquired using unmanned aerial vehicles (UAVs). The system is aimed at small-scale remote sensing applications, providing high spatial and temporal resolution observations of aquatic environments at a relatively low operational cost.

---

RIOS was developed within the *Grupo de Hidráulica Experimental* at the *Instituto de Mecánica de los Fluidos e Ingeniería Ambiental (IMFIA)*, Faculty of Engineering, Universidad de la República (UdelaR), Uruguay.
The tool was conceived to address the need for remote sensing methodologies tailored to aquatic environments using UAV platforms equipped with optical and thermal cameras. RIOS enables the estimation of hydrodynamic and water quality parameters through image-based techniques, offering a flexible and cost-effective alternative to traditional in situ monitoring and satellite-based observations.

## RIOS processing modules

RIOS is structured into three main processing modules, each addressing a specific stage of the UAV-based remote sensing workflow for water surface observation. The modules can be used independently or combined, depending on the sensor configuration and the target application.

### RIOS-G: Image Georeferencing

**RIOS-G** handles the geometric processing and georeferencing of UAV-acquired images of water surfaces. Unlike classical photogrammetric approaches that rely on Ground Control Points (GCPs) located on stable land surfaces, RIOS-G is specifically designed for environments where fixed reference points are absent or unreliable, such as rivers, lakes, reservoirs, estuaries, and coastal waters.

The module implements a *direct georeferencing* strategy based on acquisition geometry, using:

* UAV or Camera position (GNSS).
* Camera orientation (roll, pitch, yaw).
* Intrinsic camera parameters (field of view and sensor geometry).
* Simplified assumptions regarding the water surface geometry.

The outputs include spatially referenced imagery and, when applicable, extraction of pixel intensity (mono or RGB) in specified real coordinates points.

---

### RIOS-R: Reflectance-Based Processing

**RIOS-R** is dedicated to the processing and analysis of UAV imagery in the reflective spectral domain (visible and near-infrared). Its objective is to produce physically consistent reflectance products suitable for quantitative remote sensing applications over water bodies.

This module includes routines for:

* Correction of sensor-related effects (vignetting, lens distortion),
* Conversion from digital numbers to radiance and surface or apparent reflectance,
* Normalization for illumination and viewing geometry effects.
* Generation of georeferenced water surface reflectance maps.
* Extraction of water surface reflectance in specified real coordinates points.

RIOS-R enables reflectance-based applications such as water quality assessment, turbidity and suspended matter proxies, spectral indices, and surface pattern analysis. The module is designed to be sensor-agnostic and compatible with both consumer-grade and scientific multispectral cameras.

---

### RIOS-T: Thermal Processing

**RIOS-T** focuses on the processing and analysis of thermal infrared (TIR) imagery acquired from UAV platforms. The module is designed to retrieve spatially distributed surface water temperature fields and to support thermal-based hydrodynamic and environmental studies.

Core functionalities include:

* Radiometric calibration of thermal cameras,
* Correction of sensor-related effects (vignetting, non-uniformity, lens distortion),
* Correction of sensor non-uniformity and drift effects,
* Consideration of surface emissivity and viewing geometry,
* Generation of georeferenced water surface temperature maps.
* Extraction of water surface temperature in specified real coordinates points.

RIOS-T supports applications such as thermal plume detection, identification of inflows and outflows, analysis of surface thermal heterogeneity, and monitoring of river, lake, and coastal thermal dynamics.

---

## Environment setup (recommended)

This project was developed and validated using a Conda environment.
Due to GDAL, Rasterio and Shapely dependencies, Conda is strongly recommended.

```bash
conda env create -f environment.lock.yml
conda activate rios
```

## Python dependencies

A complete list of Python dependencies is also provided in *[`requirements.txt`](./requirements.txt)*
his file was automatically generated from the source code in:
* *./RIOS*
* *./core*
and is intended mainly for:
* documentation purposes
* inspection of used libraries
* non-Conda environments (not officially supported)
* 
⚠️ Installing dependencies using pip install -r requirements.txt is not recommended due to binary dependencies (GDAL, Rasterio, Shapely).
