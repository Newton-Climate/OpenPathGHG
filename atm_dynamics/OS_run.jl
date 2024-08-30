using Plots
using DataFrames
using FFTW
using Statistics
using Random

include("OS_Functions.jl")
nx = 500
ny = 500
dx = 8e-6
dy = 8e-6
Lx = 0.004
x_range = [-Lx, Lx]
y_range = [-Lx, Lx]
W_0 = 10
A_0 = 10
z = 1 # [m] beam propagation length and phase screen placement
lambda = 6.33e-7
E_0 = gen_gauss_beam(nx,ny,y_range,x_range,W_0,A_0,lambda)
# Phase screen method, repeat for n number of phase screens determined by L
C2n = 10e-15
L0  = 10
p_filter = 2.36 # 99th percentile
phase_screen =gen_phase_screen("kolmogorov", 10^(-15), 10, 0.1,500,500,2.36)
num_screens = 5
ps_num=0
if isdir("test")==false mkdir("test") end; 
loadpath = "./" * "test" * "/"; anim2 = Animation(loadpath,String[])
println("Animation directory: $(anim2.dir)")
while ps_num < num_screens
    println(ps_num)
    ps_x = (ps_num+1) * z
    B = fresnel_prop(E_0,Lx,lambda,nx,z)
    hp2 = heatmap((1:nx).*dx,(1:nx).*dx, abs.(B.^2), ylabel="Ly [m]", xlabel="Lx [m]", colorbar_size=20, right_margin = 15Plots.mm, title="Intensity [W/m^2], z= " * string(ps_x) * "m" * ", " * string(ps_num) * " Phase Screens")
    display(plot(hp2));
    frame(anim2);
    nu = B .* exp.(im .* phase_screen)
    ps_num = ps_num + 1
    hp3 = heatmap((1:nx).*dx,(1:nx).*dx, abs.(nu.^2), ylabel="Ly [m]", xlabel="Lx [m]", colorbar_size=20, right_margin = 15Plots.mm, title="Intensity [W/m^2], z= " * string(ps_x) * "m" * ", " * string(ps_num) * " Phase Screens")
    display(plot(hp3));
    frame(anim2);
    E_0 = nu
end
if ps_num == num_screens
    E_fin = fresnel_prop(E_0,Lx,lambda,nx,z)
    ps_x = num_screens + 1
    hp4 = heatmap((1:nx).*dx,(1:nx).*dx, abs.(E_fin.^2), ylabel="Ly [m]", xlabel="Lx [m]", colorbar_size=20, right_margin = 15Plots.mm, title="Intensity [W/m^2], z= " * string(ps_x) * "m" * ", " * string(ps_num) * " Phase Screens")
    display(plot(hp4));
    frame(anim2);
end
fnm_gif = "vKarman.gif"
gif(anim2, fnm_gif, fps = 5)

return E_fin

I_fin = real(E_fin.*conj(E_fin))
R_diam = 0.002
mask = R_x(R_diam,dx,nx) #enter aperture radius in m
I_mask = I_fin.*mask
heatmap(mask)
heatmap(I_fin)
A=sum(sum(abs.(I_fin)))*Lx^2
B=sum(sum(abs.(I_mask)))*pi*(R_diam/2)^2
B/A
println("Intensity at Receiver: " * string(B) * " W")
println("Intensity Reduced by " * string((B/A)*100) * "%")