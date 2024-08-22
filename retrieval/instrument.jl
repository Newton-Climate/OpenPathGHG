### This module contains the functions to simulate the open-path Instrument
# including the laser modulation, the slow sweep, the fast sweep, and the lock-in amplifier
#
## Notes for modeling the instrument measurements
    # generate the wavenumber grid from fast and slow sweepsd 
    # generate the intensity modulation of the laser light over time
    # calculate the absorbance of the laser as a result of gas absorbance
    # apply band-pass filter to isolate the 2-F signal
    # simulate the lock-in by multiplying by 2F signal
    # apply a low-pass filter to isolate the absorption line-shape
    # result should be the second derivative of absorption

using vSmartMOM.Absorption, SpectralFits
using DSP, Waveforms


"""
    struct Instrument{FT}

Instrument contains the parameters required to perform the simulation of the instrument
===============================

# Fields

- `wavenumber_range::Tuple{FT, FT}`: (cm⁻¹) The minimum and maximum wavenumbers.
- `avg_laser_intensity::FT`: (mW/m²) The average laser intensity.
- `slow_sweep_period::FT`: (s) The period of the slow sweep.
- `tuning_rate::FT`: (cm⁻¹/s) The tuning rate of the laser.
- `mod_amplitude::FT`: (cm⁻¹) The amplitude of the modulation.
- `mod_freq::FT`: (Hz) The frequency of the modulation.
- `num_samples::Int`: Number of samples in the modulation.
- `pathlength::FT`: (cm) The pathlength of the instrument.

===============================

# Example
instr = Instrument((5000.0, 5100.0), 1.0, 10.0, 0.02, 0.01, 10.0, 100, 1.0)
"""
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

"""
convert the amplitude of the fast modulation from nanometers to wave-numbers (1/cm)

Parameters
----------
min_wavenumber: float
    The minimum wavenumber
max_wavenumber: float
amp_mod_nm: float
    The amplitude of the modulation in nanometers
===============================
Returns
delta_wavenumber : float
    The amplitude of the modulation in wave-numbers
"""
function convert_amplitude(min_wavenumber::Float64,
    max_wavenumber::Float64,
    amp_mod_nm::Float64)
    # Calculate the central wavenumber
    central_wavenumber = (min_wavenumber + max_wavenumber) / 2
    # Convert amplitude from nanometers to centimeters
    amp_mod_cm = amp_mod_nm * 1e-7
    # Calculate the change in wavenumber 
    # corresponding to the amplitude
    delta_wavenumber = amp_mod_cm / ((1 / central_wavenumber)^2)

    return delta_wavenumber
end


"""
generate the triangle wave for the slow sweep
    ================================
    Parameters:
    - `peak_value`: Peak value (or amplitude) of the triangle wave
    - `slow_sweep_period`: Period of the slow sweep
    - `num_samples`: Number of samples in the slow sweep
    ================================
    Returns:
    - `triangle_wave`: The triangle wave
"""
function generate_triangle_wave(peak_value::Real,
    slow_sweep_period::Real, 
    num_samples::Int)

    # Generate the time-vector, which is slow_sweep_period long
    # t is only half a cycle 
    # for sweep of laser
    t = range(0, stop=0.5, length=num_samples)
    # Generate the triangle wave using the trianglewave function
    triangle_wave = peak_value * trianglewave1.(t)

    return triangle_wave
end


"""
generate the fast sweep.
Uses sine function
    ================================
    Parameters:
    - `mod_amplitude`: Amplitude of the fast sweep
    - `mod_freq`: Frequency of the fast sweep
    - `slow_sweep_period`: Period of the slow sweep
    - `num_samples`: Number of samples in the fast sweep
    - `phase_shift`: Phase shift of the fast sweep
    ================================
    Returns:
    - `fast_sweep`: The fast sweep
"""
function generate_fast_sweep(mod_amplitude::Real,
    mod_freq::Real,
    slow_sweep_period::Real,
    num_samples::Int;
    phase_shift::Real=0.0)

    # Calculate the number of cycles within half the triangle wave period
    num_cycles = mod_freq * slow_sweep_period / 2
    # Generate the time-vector to cover num_cycles * 2π within half the period
    t = range(0, stop=num_cycles * 2π, length=num_samples)
    # Generate the fast sweep using sine function with phase shift
    fast_sweep = mod_amplitude * sin.(t .+ phase_shift)
    
    return fast_sweep
end


"""
generate the wave-number (spectral) grid for the instrument.
This includes slow sweep (triangle wave)
and slow sweep (sine-wave)
Then, the two are added to get the wave-number grid
The result is Truncated to the rising part of the slow sweep
----> This may depend on the actual measurement duration in the observations
================================================================
# Parameters:
- `instrument`: Instrument object

# Returns:
- `wavenumber_grid`: the wave-number grid
"""
function generate_spectral_grid(instrument::Instrument)
    # Generate the slow sweep with triangle wave
    # amplitude is the max minus min wave-number
    peak_value = instrument.wavenumber_range[2] - instrument.wavenumber_range[1]
    slow_sweep = generate_triangle_wave(peak_value, instrument.slow_sweep_period, instrument.num_samples)

    # Generate the fast sweep
    fast_sweep = generate_fast_sweep(instrument.mod_amplitude, instrument.mod_freq, instrument.slow_sweep_period, instrument.num_samples)

    # Generate the wave-number grid by adding the 
    # slow and fast sweeps 
    # and starting at min_wavenumber
    starting_wavenumber = minimum(instrument.wavenumber_range)
    wavenumber_grid = slow_sweep .+ fast_sweep .+ starting_wavenumber

    # Take only the rising portion of the slow sweep
    # to-do: make this an arguement into this method
    # may depend on actual measurement in retrieval
    idx_max::Int64 = floor(Int, instrument.num_samples / 2)
    # Truncate the grid to the rising part of the slow sweep
    wavenumber_grid = wavenumber_grid[1:idx_max]

    return wavenumber_grid
end


"""
calculate the max and min wave-numbersbased on the laboratory reference points in the lab data.
Reference points are the current (in miliamps)values at which the wavenumber is known.
================================================================
# Parameters:
- `I1`: Current at the first reference point
- `I2`: Current at the second reference point
- `lambda1`: Wavenumber at the first reference point
- `lambda2`: Wavenumber at the second reference point
- `current_range`: Tuple of the min and max current values
===============================
# Returns:
- `wavenumber_range`: Tuple of the min and max wavenumbers
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
Modulations in light-intensity are due to modulation of the laser current.

================================================================

# Parameters:
- `I0`: Average laser intensity
- `i0`: Amplitude of the linear intensity modulation (IM)
- `i2`: Amplitude of the nonlinear intensity modulation (IM) (ignored for now)
- `psi1`: Phase shift for linear IM/FM (ignored for now)
- `psi2`: Phase shift for nonlinear IM/FM (ignored for now)
- `w`: Angular frequency of the modulation
- `t`: Time vector
- `transmitance`: vector of transmitance values

===============================


# Returns:
- `light_intensity`: intensity of the light over time due to laser modulation
"""
function generate_intensity(instrument, I0, i0, i2, psi1, psi2, t)
    # Calculate the intensity over time
    ω = 2π * instrument.mod_freq
    light_intensity = I0 * (1 .+ i0 * cos.(ω * t .+ psi1) .+ i2 * cos.(2 * ω * t .+ psi2))
    # return only the light corresponding 
    # to the up-sweep
    n = length(transmitance)
    
    return light_intensity[1:n]
end


"""
design a signal filter that is flexable enough to be either low-pass or band-pass filter
the resulting filter can be called as filt(digital_filter, signal)

================================================================

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

===============================

Returns
-------
digital_filter: digitalfilter
    The digital filter object
"""
function design_filter(sampling_rate::Real,
    cutoff_freq::Union{Real, Tuple{Real,Real}},
    filter_order::Int,
    filter_type::String)

    # Design the filter
    if filter_type == "lowpass"
        responsetype = Lowpass(cutoff_freq; fs=sampling_rate)
        designmethod = Butterworth(filter_order)
    
    elseif filter_type == "bandpass"
        responsetype = Bandpass(cutoff_freq[1], cutoff_freq[2]; fs=sampling_rate)
        designmethod = Butterworth(filter_order)
    
    else
        error("Filter type not supported")
    end

    return digitalfilter(responsetype, designmethod)
end


"""
simulate the lock-in amplifier

    ================================

    Parameters:
    - `direct_signal`: the direct signal
    - `instrument`: The instrument object
    - `gain`: The gain of the lock-in amplifier
    - `harmonic`: The harmonic of the lock-in amplifier

    ================================

    Returns:
    - `lockin_signal`: The lock-in signal
"""
function simulate_lockin(direct_signal::AbstractArray,
    instrument::Instrument,
    gain::Real,
    harmonic::Int)

    # create the time vector
    t = range(0, stop=instrument.slow_sweep_period, length = length(direct_signal))
    # Generate the 2F signal
     lockin_y = direct_signal .* gain .* sin.(2π * harmonic * instrument.mod_freq .* t)
     # this may be be needed later for phase-agnostic retrievals
    # lockin_x = direct_signal .* gain .* cos.(2π * harmonic * instrument.mod_freq .* t)
    # Multiply the direct signal by the 2F signal
    # lockin_signal = sqrt.(lockin_x.^2 + lockin_y.^2)

    return lockin_y
end
