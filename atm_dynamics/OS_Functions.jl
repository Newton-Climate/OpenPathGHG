function kolmogorov_psd(C2n,L0,l0,k)
    Kolm_Phi = 0.033*C2n*k.^(-11/3)
    return Kolm_Phi
end
function vonKarman_psd(C2n,L0,l0,k)
    k_m = 5.92/l0
    k_0 = 2*pi/L0
    vKarm_Phi = 0.033*C2n*exp(-k^2 / k_m^2)/((k^2 + k_0^2)^(11/6))
    return vKarm_Phi
end

function tatarski_psd(C2n,L0,l0,k)
    k_m = 5.92/l0
    ttrski_Phi = 0.033*k^(-11/3)*exp(-k^2/k_m^2)
    return ttrski_Phi
end

function mod_atm_psd(C2n,L0,l0,k)
    k_l = 3.3/l0
    k_0 = 2*pi/L0
    modatm_Phi =0.033*C2n*[1+1.802*(k/k_l)-0.254*(k/k_l)^(7/6)]*exp(-k^2/k_l^2)/((k^2+k_0^2)^(11/6))
    return modatm_Phi
end

function meshgrid(x, y, nx, ny)
    X = repeat(x', nx, 1)
    Y = repeat(y,1, ny)
    return X, Y
end
function gen_gauss_beam(nx,ny,y_range,x_range,W_0,A_0,lambda)
    x = Array(LinRange(x_range[1], x_range[2], 500))
    y = Array(LinRange(y_range[1], y_range[2], 500))
    (X, Y) = meshgrid(x, y, nx, ny)
       k = 2*pi/lambda
       E = zeros(nx, ny)
       E = (A_0*exp.(-(X.^2 + Y.^2)/W_0^2))
       return E
   end

function fresnel_prop(E,Lx,lambda,nx,z) #Convolution
    k = 2*pi/lambda
    dx = Lx/nx
     fx=-1/(2*dx):1/Lx:1/(2*dx)-1/Lx
    (FX , FY) = meshgrid(fx, fx, nx, ny)
    H = exp.((((FX.^2) + (FY.^2))*z*lambda*pi*-1*im))
    H = fftshift(H)
    U1=fft(fftshift(E))
    U2=H.*U1
    u2=ifftshift(ifft(U2))
   return u2          
end

function gen_phase_screen(turb_model::String,C2n, L0, l0,nx, ny, p_filter)
    k= []
    k = 2*pi./rand(l0:L0,nx,ny)
    PSD = [] 
    if turb_model == "kolmogorov"
        PSD =  kolmogorov_psd(C2n,L0,l0,k)
    elseif turb_model == "vonKarman"
        PSD = vonKarman_psd(C2n,L0,l0,k)
    elseif turb_model == "tatarski"
        PSD = tatarski_psd(C2n,L0,l0,k)
    elseif turb_model == "modAtm"
        PSD = mod_atm_psd(C2n,L0,l0,k)
    end
    a_r = 1/sqrt(2)*(rand(nx, ny) + 1*im*rand(nx,ny));
    phase_field = a_r.*sqrt.(PSD)/k
    phase_space = ifft(phase_field)
    std_ps = std(real(phase_space))
    mean_ps = mean(real(phase_space))
    cond1 = real(phase_space) .< mean_ps + p_filter*std_ps
    cond2 = real(phase_space) .> mean_ps - p_filter*std_ps
    result = cond1 .& cond2
    filt_ps = phase_space.*result
return filt_ps
end

function R_x(d,dx,nx)
    d = d/dx
    mask = zeros(nx,nx)
    x_range = (1:nx)
    for y in x_range
        for x in x_range
            r_d = sqrt((x - (nx/2))^2 + (y - (nx/2))^2)
                if r_d <= (d/2)
                    mask[y, x] = 1
                end
        end
    end    
    return mask
end