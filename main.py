#Estudiante: Darieth Fonseca Zuniga, Carne: B62738.
#Tarea 1, Modelos probabilisticos de senales y sistemas.
import numpy as np 
import scipy
from scipy import signal
from scipy import integrate
from scipy import stats 
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm,rayleigh
from pylab import plot,show,hist,title
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

#Lectura de datos
datos = pd.read_csv('bits10k.csv')
#matriz de datos
matriz = datos.to_numpy()

#Frecuencia de operación
f = 10000 #Hz
#Duración del período de la onda de cada símbolo
T = 1/f #Período de cada símbolo
#Número de puntos de muestreo por período
p = 100
#Punto de muestreo para cada periodo
tp = np.linspace(0,T,p)
#Frecuencia de muestreo
fm = p/T
#Creación de la linea temporal Tx
t = np.linspace(0,10000*T,10000*p)
#Vector de la señal
senal = np.zeros(t.shape)

#Punto 1.
#Forma de onda de la portadora
sin = np.sin(2*np.pi*f*tp)
#Creación de la señal modulada BPSK
for k, b in enumerate(matriz):
  if b == 1:
    senal[k*p:(k+1)*p] = b * sin
  else:
    senal[k*p:(k+1)*p] = -1 * sin
#Visualizacion de la modulación, bits enviados
pb = 5 #número de pruebas
plt.figure()
plt.plot(senal[0:5*p])
plt.xlabel('Puntos de prueba.')
plt.ylabel('Señal.')
plt.savefig('Tx')

#Punto 2.
#Potencia promedio, usando la defición matemática
# Potencia instantánea
pin = senal**2
# Potencia promedio 
pp = integrate.trapz(pin, t) / (10000 * T)
print('La potencia promedio es: ', pp, 'W.')

#Punto 3.
#Crear Ruido
#-2dB
SNR = -2#Relación señal a ruido deseada
pn = pp / (10**(SNR / 10))#Potencia del ruido
sigma = np.sqrt(pn)#Desviación estándar del ruido
ruido = np.random.normal(0, sigma, senal.shape)#Ruido gaussiano
#Simular el canal
Rx0 = senal + ruido
#Visualización de los primeros bits recibidos
plt.figure()
plt.plot(Rx0[0:5*p])
plt.xlabel('Puntos de prueba.')
plt.ylabel('Señal.')
plt.savefig('Rx_-2dB')
#-1dB
SNR = -1
pn = pp / (10**(SNR / 10))#Potencia del ruido
sigma = np.sqrt(pn)#Desviación estándar del ruido
ruido = np.random.normal(0, sigma, senal.shape)#Ruido gaussiano
#Simular el canal
Rx1 = senal + ruido
#Visualización de los primeros bits recibidos
plt.figure()
plt.plot(Rx1[0:5*p])
plt.xlabel('Puntos de prueba.')
plt.ylabel('Señal.')
plt.savefig('Rx_-1dB')
#0dB
SNR = 0
pn = pp / (10**(SNR / 10))#Potencia del ruido
sigma = np.sqrt(pn)#Desviación estándar del ruido
ruido = np.random.normal(0, sigma, senal.shape)#Ruido gaussiano
#Simular el canal
Rx2 = senal + ruido
#Visualización de los primeros bits recibidos
plt.figure()
plt.plot(Rx2[0:5*p])
plt.xlabel('Puntos de prueba.')
plt.ylabel('Señal.')
plt.savefig('Rx_0dB')
#1dB
SNR = 1
pn = pp / (10**(SNR / 10))#Potencia del ruido
sigma = np.sqrt(pn)#Desviación estándar del ruido
ruido = np.random.normal(0, sigma, senal.shape)#Ruido gaussiano
#Simular el canal
Rx3 = senal + ruido
#Visualización de los primeros bits recibidos
plt.figure()
plt.plot(Rx3[0:5*p])
plt.xlabel('Puntos de prueba.')
plt.ylabel('Señal.')
plt.savefig('Rx_1dB')
#2dB
SNR = 2
pn = pp / (10**(SNR / 10))#Potencia del ruido
sigma = np.sqrt(pn)#Desviación estándar del ruido
ruido = np.random.normal(0, sigma, senal.shape)#Ruido gaussiano
#Simular el canal
Rx4 = senal + ruido
#Visualización de los primeros bits recibidos
plt.figure()
plt.plot(Rx4[0:5*p])
plt.xlabel('Puntos de prueba.')
plt.ylabel('Señal.')
plt.savefig('Rx_2dB')
#3dB
SNR = 3
pn = pp / (10**(SNR / 10))#Potencia del ruido
sigma = np.sqrt(pn)#Desviación estándar del ruido
ruido = np.random.normal(0, sigma, senal.shape)#Ruido gaussiano
#Simular el canal
Rx5 = senal + ruido
#Visualización de los primeros bits recibidos
plt.figure()
plt.plot(Rx5[0:5*p])
plt.xlabel('Puntos de prueba.')
plt.ylabel('Señal.')
plt.savefig('Rx_3dB')

#Punto 4.
# Antes del canal ruidoso
fw, PSD = signal.welch(senal, fm, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig('Densidad_antes_del_canal')
# Después del canal ruidoso
#SNR = -2 dB
fw, PSD = signal.welch(Rx0, fm, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig('Densidad_SNR=-2')
#SNR = -1 dB
fw, PSD = signal.welch(Rx1, fm, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig('Densidad_SNR=-1')
#SNR = 0 dB
fw, PSD = signal.welch(Rx2, fm, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig('Densidad_SNR=0')
#SNR = 1 dB
fw, PSD = signal.welch(Rx3, fm, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig('Densidad_SNR=1')
#SNR = 2 dB
fw, PSD = signal.welch(Rx4, fm, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig('Densidad_SNR=2')
#SNR = 3 dB
fw, PSD = signal.welch(Rx5, fm, nperseg=1024)
plt.figure()
plt.semilogy(fw, PSD)
plt.xlabel('Frecuencia / Hz')
plt.ylabel('Densidad espectral de potencia / V**2/Hz')
plt.savefig('Densidad_SNR=3')


#Punto 5
# Pseudo-energía de la onda original (esta es suma, no integral)
Es = np.sum(sin**2)
# Inicialización del vector de bits recibidos

#Caso SNR = -2 db
bitsRx0 = np.zeros(matriz.shape)
# Decodificación de la señal por detección de energía
for k, b in enumerate(matriz):
    Ep = np.sum(Rx0[k*p:(k+1)*p] * sin)
    if Ep > Es/2:
        bitsRx0[k] = 1
    else:
        bitsRx0[k] = 0

err = np.sum(np.abs(matriz - bitsRx0))
BER = err/10000
print('Hay un total de {} errores en {} bits para una tasa de error de {} con SNR = -2dB.'.format(err, 10000, BER))

#Caso SNR = -1 db
bitsRx1 = np.zeros(matriz.shape)
# Decodificación de la señal por detección de energía
for k, b in enumerate(matriz):
    Ep = np.sum(Rx1[k*p:(k+1)*p] * sin)
    if Ep > Es/2:
        bitsRx1[k] = 1
    else:
        bitsRx1[k] = 0

err = np.sum(np.abs(matriz - bitsRx1))
BER = err/10000
print('Hay un total de {} errores en {} bits para una tasa de error de {} con SNR = -1dB.'.format(err, 10000, BER))

#Caso SNR = 0 db
bitsRx2 = np.zeros(matriz.shape)
# Decodificación de la señal por detección de energía
for k, b in enumerate(matriz):
    Ep = np.sum(Rx2[k*p:(k+1)*p] * sin)
    if Ep > Es/2:
        bitsRx2[k] = 1
    else:
        bitsRx2[k] = 0

err = np.sum(np.abs(matriz - bitsRx2))
BER = err/10000
print('Hay un total de {} errores en {} bits para una tasa de error de {} con SNR = 0dB.'.format(err, 10000, BER))

#Caso SNR = 1 db
bitsRx3 = np.zeros(matriz.shape)
# Decodificación de la señal por detección de energía
for k, b in enumerate(matriz):
    Ep = np.sum(Rx3[k*p:(k+1)*p] * sin)
    if Ep > Es/2:
        bitsRx3[k] = 1
    else:
        bitsRx3[k] = 0

err = np.sum(np.abs(matriz - bitsRx3))
BER = err/10000
print('Hay un total de {} errores en {} bits para una tasa de error de {} con SNR = 1dB.'.format(err, 10000, BER))

#Caso SNR = 2 db
bitsRx4 = np.zeros(matriz.shape)
# Decodificación de la señal por detección de energía
for k, b in enumerate(matriz):
    Ep = np.sum(Rx4[k*p:(k+1)*p] * sin)
    if Ep > Es/2:
        bitsRx4[k] = 1
    else:
        bitsRx4[k] = 0

err = np.sum(np.abs(matriz - bitsRx4))
BER = err/10000
print('Hay un total de {} errores en {} bits para una tasa de error de {} con SNR = 2dB.'.format(err, 10000, BER))

#Caso SNR = 3 db
bitsRx5 = np.zeros(matriz.shape)
# Decodificación de la señal por detección de energía
for k, b in enumerate(matriz):
    Ep = np.sum(Rx5[k*p:(k+1)*p] * sin)
    if Ep > Es/2:
        bitsRx5[k] = 1
    else:
        bitsRx5[k] = 0

err = np.sum(np.abs(matriz - bitsRx5))
BER = err/10000
print('Hay un total de {} errores en {} bits para una tasa de error de {} con SNR = 3dB.'.format(err, 10000, BER))

#Punto 6
#Valores obtenidos en consola
BER = [0,0,0,0,0,0]
SNR = [-2,-1,0,1,2,3]
#Gráfica Ber vs SNR
plt.figure()
plt.title('BERvsSNR')
plt.ylabel('BER')
plt.xlabel('SNR')
plt.plot(SNR,BER)
plt.savefig("BERvsSNR")
