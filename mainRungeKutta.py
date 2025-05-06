#   Codigo que implementa el metodo de Runge-Kutta
#   de cuarto orden para resolver una ecuacion diferencial
#   
#
#           Autor:
#   Dr. Ivan de Jesus May-Cen
#   imaycen@hotmail.com
#   Version 1.0 : 06/05/2025
#

import numpy as np
import matplotlib.pyplot as plt

# Definición de la EDO: dy/dx = f(x, y)
def f(x, y):
    return x * np.sqrt(y)  # Puedes cambiar esta función

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4(f, x0, y0, x_end, h):
    x_vals = [x0]
    y_vals = [y0]
    
    x = x0
    y = y0
    
    print(f"{'x':>10} {'y':>15}")
    print(f"{x:10.4f} {y:15.6f}")
    
    while x < x_end:
        k1 = f(x, y)
        k2 = f(x + h/2, y + h/2 * k1)
        k3 = f(x + h/2, y + h/2 * k2)
        k4 = f(x + h, y + h * k3)
        
        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
        
        x_vals.append(x)
        y_vals.append(y)
        
        print(f"{x:10.4f} {y:15.6f}")
    
    return x_vals, y_vals

# Parámetros iniciales
x0 = 0
y0 = 1
x_end = 2
h = 0.1

# Llamada al método de Runge-Kutta
x_vals, y_vals = runge_kutta_4(f, x0, y0, x_end, h)

# Graficar la solución
plt.figure(figsize=(8,5))
plt.plot(x_vals, y_vals, 'bo-', label="Solución RK4")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Método de Runge-Kutta de 4to Orden")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("runge_kutta4.png")
plt.show()

