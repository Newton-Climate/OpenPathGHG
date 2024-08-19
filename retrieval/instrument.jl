using LinearAlgebra, Plots, Interpolations, Statistics
using vSmartMOM.Absorption, SpectralFits
using DataFrames, Peaks, CSV, DSP
using Waveforms


## Notes for modeling the instrument measurements
    # generate the wavenumber grid 
    # generate the intensity of the light over time
    # calculate the absorbance of the light over time i.e., the measurement grid
    # apply band-pass filter to isolate the 2-F signal
    # simulate the lock-in by multiplying by 2F signal
    # apply a low-pass filter to get the lock-in signal
    # result should be the second derivative of absorption 


# struct to store laser parameters used to generate wave-number grid
Base.@kwdef struct Instrument{FT}
    wavenumber_range::Tuple{FT, FT}
    avg_laser_intensity::FT
    slow_sweep_period::FT
    tuning_rate::FT
    mod_amplitude::FT
    mod_freq::FT
    num_samples::Int
    pathlength::FT
end


function convert_amplitude(min_wavenumber::Float64, max_wavenumber::Float64, amp_mod_nm::Float64)
    # Calculate the central wavenumber
    central_wavenumber = (min_wavenumber + max_wavenumber) / 2
    # Convert amplitude from nanometers to centimeters
    amp_mod_cm = amp_mod_nm * 1e-7
    # Calculate the change in wavenumber corresponding to the amplitude
    delta_wavenumber = amp_mod_cm / ((1 / central_wavenumber)^2)

    return delta_wavenumber
end


function generate_triangle_wave(peak_value::Real,
    slow_sweep_period::Real, 
    num_samples::Int)

    # Generate the time-vector, which is slow_sweep_period long
    # normalize t to only 0.5 cycles
    t = range(0, stop=0.5, length=num_samples)

    # Normalize t to represent time over one full cycle
    # t = t ./ slow_sweep_period

    # Generate the triangle wave using the trianglewave function
    triangle_wave = peak_value * trianglewave1.(t)

    @show triangle_wave[1]
    @show maximum(triangle_wave)
    @show argmax(triangle_wave)
    
    return triangle_wave
end



"""generate the fast sweep.
Uses sine function"""
function generate_fast_sweep(mod_amplitude, mod_freq, slow_sweep_period, num_samples; phase_shift=0.0)
    # Calculate the number of cycles within half the triangle wave period
    num_cycles = mod_freq * slow_sweep_period / 2
    
    # Generate the time-vector to cover num_cycles * 2π within half the period
    t = range(0, stop=num_cycles * 2π, length=num_samples)
    
    # Generate the fast sweep using sine function with phase shift
    fast_sweep = mod_amplitude * sin.(t .+ phase_shift)
    
    return fast_sweep
end


"""
generate the wave-number grid for the instrument.
This includes slow sweep (triangle wave)
and slow sweep (sine-wave)
"""
function generate_spectral_grid(instrument::Instrument)
    # Generate the slow sweep with triangle wave
    peak_value = instrument.wavenumber_range[2] - instrument.wavenumber_range[1]
    slow_sweep = generate_triangle_wave(peak_value, instrument.slow_sweep_period, instrument.num_samples)
    # Generate the fast sweep
    fast_sweep = generate_fast_sweep(instrument.mod_amplitude, instrument.mod_freq, instrument.slow_sweep_period, instrument.num_samples)
    # Generate the wave-number grid by adding the two sweeps and starting at min_wavenumber
    min_wavenumber = minimum(instrument.wavenumber_range)
    wavenumber_grid = slow_sweep .+ fast_sweep .+ min_wavenumber
    @show length(wavenumber_grid)
    # Take only the rising portion of the slow sweep
    idx_max::Int64 = floor(Int, instrument.num_samples / 2)
    @show idx_max
    # Truncate the grid to the rising part of the slow sweep
    wavenumber_grid = wavenumber_grid[1:idx_max]
    @show maximum(wavenumber_grid)
    @show argmax(wavenumber_grid)

    return wavenumber_grid
end

"""
calculate the max and min wave-numbersbased on the reference points in the lab data
"""
function calc_wavenumber_range(I1, I2, lambda1, lambda2, current_range)
    # get the min and max current_range
        min_current = minimum(current_range)
        max_current = maximum(current_range)
    # Calculate the slope
    slope = (lambda2 - lambda1) / (I2 - I1)
    
    # Calculate the start and end wavelengths    
    lambda_start = slope * (min_current - I2) + lambda2
    lambda_end = slope * (max_current - I2) + lambda2
    
    # Return the max and min wavenumbers 
    return [lambda_start, lambda_end]
end


"""
Generate the intensity of the light over time.

# Parameters:
- `I0`: Average laser intensity
- `i0`: Amplitude of the linear intensity modulation (IM)
- `i2`: Amplitude of the nonlinear intensity modulation (IM) (ignored for now)
- `psi1`: Phase shift for linear IM/FM (ignored for now)
- `psi2`: Phase shift for nonlinear IM/FM (ignored for now)
- `w`: Angular frequency of the modulation
- `t`: Time vector
- `transmitance`: vector of transmitance values

# Returns:
- `data_mult`: Scaled intensity of the light over time
"""
function generate_intensity(instrument, I0, i0, i2, psi1, psi2, t, transmitance)
    t = range(0, stop=instrument.slow_sweep_period, length=instrument.num_samples)
    # Calculate the intensity over time
    I0_t = I0 * (1 .+ i0 * cos.(w * t .+ psi1) .+ i2 * cos.(2 * w * t .+ psi2))
    
    # Scale by the intensity of the light
    data_mult = transmitance .* I0_t
    
    return data_mult
end




"""
design a signal filter that is flexable enough to be either low-pass or band-pass filter
the resulting filter can be called as filt(digital_filter, signal)

Parameters
----------
sampling_rate: float
    The sampling rate of the signal
cutoff_freq: float or Tuple
the cut-off frequency or frequencies of the filter
filter_order: int
    The order of the filter
filter_type: str
the type of filter: either lowpass or bandpass

Returns
-------
digital_filter: digitalfilter
    The digital filter object
"""
function design_filter(sampling_rate, cutoff_freq, filter_order, filter_type)
    # Calculate the normalized cutoff frequency
    cutoff_freq_norm = cutoff_freq / (sampling_rate / 2)
    
    # Design the filter
    if filter_type == "lowpass"
        responsetype = Lowpass(cutoff_freq; fs=sampling_rate)
        designmethod = Butterworth(filter_order)
    
    elseif filter_type == "bandpass"
        responsetype = Bandpass(cutoff_freq; fs=sampling_rate)
        designmethod = Butterworth(filter_order)
    
    else
        error("Filter type not supported")
    end

    # Instantiate the digital filter
    digital_filter = digitalfilter(responsetype, designmethod)
    
    return digital_filter
end

### read the dataset
#laser_data = "data/lab_data_02-23/laser/"

#df = CSV.read(laser_data*"/ch4_meas_0_1.txt", DataFrame, delim=',', header=["cell", "room", "lockin"])

# filter out the rising parts of the wavelength sweep
#(lockin_rise, environmental_rise) = find_peaks(df)
# normalize the lockin-in signal to the environmental signal
# normalized_lockin = normalize_lockin_signal(environmental_rise, lockin_rise)


### Define the instrument parameters
I_end = 45.1
I_start = 30.1
I1 = 36.6
I2 = 37.6

lambda1 = 1653.678
lambda2 = 1653.707
lambda_start, lambda_end = calc_wavenumber_range(I1, I2, lambda1, lambda2, (I_start, I_end))
# convert nanometers to wave-numbers (1/cm) of lambda_start:lambda_end
min_wavenumber = 1e7 / lambda_end
max_wavenumber = 1e7 / lambda_start

wavenumber_range = (min_wavenumber, max_wavenumber)

mod_amplitude = 0.00367 # experimentally derived parameter for fast sweep 
# convert mod_amplitude from nanometers to 1/cm

mod_amplitude = convert_amplitude(min_wavenumber, max_wavenumber, mod_amplitude)
@show mod_amplitude
mod_freq = 10.0e3 # Hz
sampling_rate = 100000
num_samples = 2*250517
slow_sweep_period = num_samples*2/sampling_rate #sec per measurement sweep 
tuning_rate = (max_wavenumber - min_wavenumber) / slow_sweep_period
pathlength= 1e5



I0 = 4.0 # avg laser intensity
i0 = 0.00005 # amplitude of the linear IM
i2 = 0 # ignore the nonlinearities for the moment
psi1 = 0 # ignore phase shift rn
psi2 = 0 # ignore phase shift rn
w = 2 * pi * mod_freq # example angular frequency
t = range(0, stop=slow_sweep_period, length=num_samples)

# create the instrument object
instrument = Instrument(wavenumber_range=wavenumber_range, mod_freq=mod_freq,
mod_amplitude=mod_amplitude, slow_sweep_period=slow_sweep_period,
num_samples=num_samples, pathlength=pathlength,
avg_laser_intensity=I0, tuning_rate=tuning_rate
)

# generate the spectral grid
grid = generate_spectral_grid(instrument)
@show grid[end]

data_mult = generate_intensity(instrument, I0, i0, i2, psi1, psi2, t, 1e3)