import numpy as np
from scipy.optimize import least_squares

def fit_gaussian(DT):

    x = np.arange(DT.shape[1])
    y = np.arange(DT.shape[0])
    
    X, Y = np.meshgrid(x, y)
    
    def gaussian_func(params, x, y):
        x0, sigx, y0, sigy, Ax,Ay = params
        return Ax * np.exp(-((x - x0) ** 2 / (2 * sigx ** 2)))+ Ay * np.exp(- ((y - y0) ** 2 / (2 * sigy ** 2)))
    
    def objective_func(params, x, y, data):
        model = gaussian_func(params, x, y)
        residuals = model - data
        return residuals.ravel()
    
    initial_params = [DT.shape[0]/2 , 0.5, DT.shape[1] / 2, 0.5, 1,1]
     
    result = least_squares(objective_func, initial_params, args=(X, Y, DT))

    # Parámetros ajustados
    optimized_params = result.x

    # Imagen ajustada por el paraboloide
    DT_ajustada = gaussian_func(optimized_params, X, Y)
    
    return DT_ajustada

    
def fit_paraboloide(DT):
    
    x = np.arange(DT.shape[1])
    y = np.arange(DT.shape[0])
    
    X, Y = np.meshgrid(x, y)
    
    def parabolic_func(params, x, y):
        a,b,c,d,e,f = params
        return a * x**2 + b*x + c + d * y**2 + e*y +f
    
    def objective_func(params, x, y, data):
        model = parabolic_func(params, x, y)
        residuals = model - data
        return residuals.ravel()
    
    initial_params = [0, 0, 0, 0, 0,0]
     
    DT_flat=DT
    
    result = least_squares(objective_func, initial_params, args=(X, Y, DT_flat)) #,verbose=2)

    # Parámetros ajustados

    optimized_params = result.x

    # Imagen ajustada por el paraboloide
    DT_ajustada = parabolic_func(optimized_params, X, Y)
    
    return DT_ajustada

