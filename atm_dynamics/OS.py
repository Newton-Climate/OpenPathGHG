def kolmogorov_psd(C2n, L0, l0, k):
    Kolm_Phi = 0.033 * C2n * k**(-11/3)
    return Kolm_Phi

def vonKarman_psd(C2n, L0, l0, k):
    k_m = 5.92 / l0
    k_0 = 2 * np.pi / L0
    vKarm_Phi = 0.033 * C2n * np.exp(-k**2 / k_m**2) / ((k**2 + k_0**2)**(11/6))
    return vKarm_Phi

def tatarski_psd(C2n, L0, l0, k):
    k_m = 5.92 / l0
    ttrski_Phi = 0.033 * k**(-11/3) * np.exp(-k**2 / k_m**2)
    return ttrski_Phi

def mod_atm_psd(C2n, L0, l0, k):
    k_l = 3.3 / l0
    k_0 = 2 * np.pi / L0
    modatm_Phi = 0.033 * C2n * (1 + 1.802 * (k / k_l) - 0.254 * (k / k_l)**(7/6)) * np.exp(-k**2 / k_l**2) / ((k**2 + k_0**2)**(11/6))
    return modatm_Phi

def meshgrid(x, y, nx, ny):
    X, Y = np.meshgrid(np.repeat(x, nx), np.repeat(y, ny))
    return X, Y

def gen_gauss_beam(nx, ny, y_range, x_range, W_0, A_0, wavelength):
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)
    k = 2 * np.pi / wavelength
    E = A_0 * np.exp(-(X**2 + Y**2) / W_0**2)
    return E

def gen_gauss_long(nx, ny, y_range, x_range, W_0, A_0, L, z, wavelength):
    kappa = 2*np.pi/wavelength
    beam_div = 5e-4
    F_0 = 1.654e-6 / (2 * beam_div)
    Theta_0 = 1-(L/F_0)
    Amps_0 = 2*L/(kappa*W_0**2)
    F = (F_0 * (Theta_0**2 + Amps_0**2) * (Theta_0 - 1)) / (Theta_0**2 + Amps_0**2 - Theta_0)
    W = W_0 * np.sqrt(Theta_0**2 + Amps_0**2)
    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    E = (A_0/np.sqrt(Theta_0**2 + Amps_0**2))*np.exp(-r**2 / W**2)*np.exp(1j*kappa*L - 1j*math.atan(Amps_0/Theta_0) - 1j * kappa * r**2/(2*F))
    return E

def fresnel_prop(E, Lx, wavelength, nx, z):
    k = 2 * np.pi / wavelength
    dx = Lx / nx
    # fx = np.fft.fftfreq(nx, d=dx)  # Frequency coordinates
    fx = np.arange(-1/(2*dx),(1/(2*dx)-1/Lx)+1/Lx,1/Lx)  # Frequency coordinates
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    H = np.fft.fftshift(H)
    U1 = np.fft.fft2(np.fft.fftshift(E))
    U2 = H * U1
    u2 = np.fft.ifftshift(np.fft.ifft2(U2))
    return u2 

def gen_phase_screen(turb_model, C2n, L0, l0, nx, ny, p_filter):
    # k = 2 * np.pi /np.random.normal(l0,L0)
    k = 2 * np.pi /l0
    if turb_model == "kolmogorov":
        PSD = kolmogorov_psd(C2n, L0, l0, k)
    elif turb_model == "vonKarman":
        PSD = vonKarman_psd(C2n, L0, l0, k)
    elif turb_model == "tatarski":
        PSD = tatarski_psd(C2n, L0, l0, k)
    elif turb_model == "modAtm":
        PSD = mod_atm_psd(C2n, L0, l0, k)
    else:
        raise ValueError("Invalid turbulence model")
        
    a_r = np.random.normal(loc=0.0, scale=1.0, size=(nx, ny))
    phase_field = a_r * np.sqrt(PSD)
    phase_space = np.fft.ifft2(phase_field)
    filt_ps = phase_space
    plt.figure()
    plt.figure(figsize=(8, 7))
    plt.imshow(np.real(filt_ps), cmap='gist_stern')
    plt.colorbar()
    plt.title('Re(Phase Screen)')
    plt.xlabel('Lx [m]')
    plt.ylabel('Ly [m]')
    plt.xticks(ticks=tick_locations, labels=tick_labels)
    plt.yticks(ticks=tick_locations, labels=tick_labels)
    plt.savefig(f'/Users/kmagno/Documents/OS_figs/Phase_Screen.png', format='png')
    # Show the plot
    plt.show()
    plt.close()
    return filt_ps

def irr_calc(EF,c,eps_0):
        # Calculate result as the element-wise squared magnitude of EF
    result = np.conj(EF) * EF
    
    # Compute Irr using vectorized operations
    Irr = np.real(result * c * eps_0 * 0.5)
    
    return Irr

def R_x(d, dx, nx):
    d = d / dx
    mask = np.zeros((nx, nx))
    x_range = np.arange(nx)
    for y in x_range:
        for x in x_range:
            r_d = np.sqrt((x - (nx / 2)) ** 2 + (y - (nx / 2)) ** 2)
            if r_d <= (d / 2):
                mask[y, x] = 1
    return mask


def plot3d_phase(phase_screen):
    # Create a 2D matrix
    matrix = np.real(phase_screen*np.conj(phase_screen))

    # Create a figure and an axes object
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Generate X and Y coordinates for the surface plot
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    X, Y = np.meshgrid(x, y)

    # Plot the surface
    surf = ax.plot_surface(X, Y, matrix, cmap='gist_stern')


    # Set labels for the axes
    ax.set_xlabel('Lx')
    ax.set_ylabel('Ly')
    ax.set_zlabel('|Phase Screen|^2')
    cbar = plt.colorbar(surf,orientation='horizontal', pad=0.2)
    plt.xticks(ticks=tick_locations, labels=tick_labels)
    plt.yticks(ticks=tick_locations, labels=tick_labels)
    plt.savefig(f'/Users/kmagno/Documents/OS_figs/Phase_Screen.png', format='png')
    # Show the plot
    plt.show()
    plt.close()
    
    
    
    
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
import math
import csv

nx = 256
ny = 256
Lx = 0.005
dx = Lx/nx
dy = Lx/ny
#x_range = [-Lx, Lx]
#y_range = [-Lx, Lx]
x_range = [-Lx, Lx]
y_range = [-Lx, Lx]
# Define the tick locations and labels
num_ticks = 5
tick_locations = np.linspace(0, nx-1, num_ticks)
tick_labels = np.linspace(0, Lx, num_ticks)

file_names = []
max_P = 0.07
W_0 = 3e-3#5e-3
I_max = max_P/(np.pi*(W_0/2)**2)
c = 3e8
eps_0 = 8.85e-12
A_0 = np.sqrt(2*I_max/(c*eps_0))

wavelength = 6.33e-7
#1.654e-6
C2n = 10e-15
L0 = 10
l0 = 0.1
turb_model = "vonKarman"
p_filter = 2.36
I_trend = []
phase_screen = gen_phase_screen(turb_model, C2n, L0, l0, nx, ny, p_filter)
plt.figure()


E_0 = np.array(gen_gauss_beam(nx, ny, y_range, x_range, W_0, A_0, wavelength))
# print('I E_0 = ')
# print(np.mean(np.real(abs(E_0*np.conj(E_0)))))
#I_trend.append(np.mean(np.real(abs(E_0*np.conj(E_0)))))
Irr_E0 = irr_calc(E_0,c,eps_0)
print('OG = ' + str(np.max(Irr_E0)))
plt.figure()
plt.figure(figsize=(8, 7))
plt.imshow(Irr_E0, cmap='gist_stern')
plt.colorbar()
plt.title('Initial Gaussian Beam, Irradiance W/m^2')
plt.xlabel('Lx [m]')
plt.ylabel('Ly [m]')
plt.xticks(ticks=tick_locations, labels=tick_labels)
plt.yticks(ticks=tick_locations, labels=tick_labels)
plt.savefig(f'/Users/kmagno/Documents/OS_figs/Initial_Irr.png', format='png')
plt.show()
plt.close() 
E_f = E_0
tot_Z = 10 # m
d_ps = 1e-2 # m, distance between each phase screen
num_screens = math.floor(tot_Z/d_ps)
print('num_screens = ' + str(num_screens))
ps_num = 0
while ps_num < num_screens:
    E = np.array(fresnel_prop(E_f, Lx, wavelength, nx, d_ps))
    ps_num = ps_num + 1
    nu = E * np.exp(1j * phase_screen)
    Irr_nu = irr_calc(nu,c,eps_0)
    if ps_num % 200 == 0:
        print(ps_num)
        plt.figure()
        plt.figure(figsize=(8, 7))
        plt.imshow((Irr_nu), cmap='gist_stern')
        plt.colorbar()
        plt.xticks(ticks=tick_locations, labels=tick_labels)
        plt.yticks(ticks=tick_locations, labels=tick_labels)
        plt.xlabel('Lx [m]')
        plt.ylabel('Ly [m]')
        plt.title('Irradiance [W/m2]\n' + 'z = ' + str(d_ps * ps_num) + 'm\n' + str(ps_num) + ' ' + str(turb_model) + ' Phase Screens')
        file_name = f'/Users/kmagno/Documents/OS_figs/Irr_{ps_num}.png'
        file_names.append(file_name)
        plt.savefig(file_name, format='png')
        plt.show()
        plt.close()  # Close the figure to free up memory
        # save array
        filename = f'/Users/kmagno/Documents/OS_csv/E_{ps_num}.csv'
        # Write the data to the CSV file
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(np.array(nu))
    E_f = nu
R_diam = 0.0025 # [m]
mask = np.zeros((nx, ny))
mask = R_x(R_diam,dx,nx)
Irr_fin = irr_calc(nu,c,eps_0)
I_mask = Irr_fin * mask
plt.figure()
plt.figure(figsize=(8, 7))
plt.imshow((I_mask), cmap='gist_stern',vmin=0, vmax=3.5)
plt.colorbar()
plt.xticks(ticks=tick_locations, labels=tick_labels)
plt.yticks(ticks=tick_locations, labels=tick_labels)
plt.xlabel('Lx [m]')
plt.ylabel('Ly [m]')
plt.title('Irradiance [W/m2] at Receiver \n' + 'R Area = ' + str(round(np.pi*(R_diam/2)**2,5)) + 'm^2\n' + 'z = ' + str(d_ps * ps_num) + 'm\n' + str(ps_num) + ' ' + str(turb_model) + ' Phase Screens')
plt.savefig(f'/Users/kmagno/Documents/OS_figs/FINAL.png', format='png')
plt.show()
plt.close()  # Close the figure to free up memory
#tot_P = np.mean(np.mean(I_mask))/(np.pi*(R_diam/2)**2)
#print('Average Power on Receiver: ' + str(tot_P) + ' W')
totP_IC= (sum(sum(Irr_fin))*Lx**2)
totP_Mask= (sum(sum(I_mask))*np.pi*(R_diam/2)**2)
print('Average Power Before Receiver: ' + str(totP_IC) + ' W')
print('Average Power on Receiver: ' + str(totP_Mask) + ' W')

from PIL import Image

# # Load the images
images = [Image.open(file_name) for file_name in file_names]

# # Create the GIF
gif_file_name = "animation_cn2_15.gif"
images[0].save(gif_file_name, save_all=True, append_images=images[1:], duration=5, loop=0)


