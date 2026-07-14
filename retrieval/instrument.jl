### Utilities for simulating an open-path wavelength-modulation spectroscopy instrument.
#
# The workflow implemented in this file is:
# 1) Build the instrument spectral grid from slow and fast sweeps.
# 2) Apply gas transmission to the laser signal.
# 3) Isolate the 2f component with a band-pass filter.
# 4) Demodulate with a lock-in reference.
# 5) Smooth with a low-pass filter to recover the line-shape.

using vSmartMOM.Absorption, SpectralFits
using DSP, Waveforms


"""
    Instrument{FT}

Parameter container for the open-path instrument simulation.

# Fields
- `wavenumber_range::Tuple{FT, FT}`: Minimum and maximum wavenumbers (cm^-1).
- `avg_laser_intensity::FT`: Mean laser intensity.
- `slow_sweep_period::FT`: Period of the slow sweep (s).
- `tuning_rate::FT`: Laser tuning rate (cm^-1 s^-1).
- `mod_amplitude::FT`: Fast modulation amplitude (cm^-1).
- `mod_freq::FT`: Fast modulation frequency (Hz).
- `num_samples::Int`: Number of samples per simulated sweep.
- `pathlength::FT`: Optical pathlength (cm).
- `sampling_rate::FT`: Acquisition sampling rate (Hz).
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
    sampling_rate::FT
end


"""
    convert_amplitude(min_wavenumber, max_wavenumber, amp_mod_nm)

Convert a modulation amplitude from nanometers to wavenumber units (cm^-1)
around the central wavenumber of the scan range.

# Arguments
- `min_wavenumber::Float64`: Minimum wavenumber (cm^-1).
- `max_wavenumber::Float64`: Maximum wavenumber (cm^-1).
- `amp_mod_nm::Float64`: Modulation amplitude in nanometers.

# Returns
- `delta_wavenumber::Float64`: Modulation amplitude in cm^-1.
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
    generate_triangle_wave(peak_value, slow_sweep_period, num_samples)

Generate the slow-sweep triangle waveform used for wavelength scanning.

# Arguments
- `peak_value::Real`: Triangle-wave peak amplitude in cm^-1.
- `slow_sweep_period::Real`: Slow-sweep period in seconds (kept for API consistency).
- `num_samples::Int`: Number of samples.

# Returns
- `triangle_wave`: Sampled slow-sweep waveform.
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
    generate_fast_sweep(mod_amplitude, mod_freq, slow_sweep_period, num_samples; phase_shift=0.0)

Generate the sinusoidal fast modulation waveform.

# Arguments
- `mod_amplitude::Real`: Fast modulation amplitude in cm^-1.
- `mod_freq::Real`: Modulation frequency in Hz.
- `slow_sweep_period::Real`: Slow-sweep period in seconds.
- `num_samples::Int`: Number of samples.
- `phase_shift::Real=0.0`: Optional phase offset in radians.

# Returns
- `fast_sweep`: Sampled fast modulation waveform.
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
    generate_spectral_grid(instrument)

Generate the instrument wavenumber grid by combining slow and fast sweeps.

The returned grid is truncated to the rising half of the slow sweep.

# Arguments
- `instrument::Instrument`: Instrument configuration.

# Returns
- `wavenumber_grid`: Simulated measurement grid in cm^-1.
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

    # Keep only the rising branch of the sweep.
    # This could be made configurable if acquisition timing changes.
    idx_max::Int64 = floor(Int, instrument.num_samples / 2)
    # Truncate the grid to the rising part of the slow sweep
    wavenumber_grid = wavenumber_grid[1:idx_max]

    return wavenumber_grid
end


"""
    calc_wavenumber_range(I1, I2, lambda1, lambda2, current_range)

Estimate scan-endpoint wavenumbers from two current-to-wavenumber reference points.

# Arguments
- `I1`: Current at the first reference point.
- `I2`: Current at the second reference point.
- `lambda1`: Wavenumber at `I1`.
- `lambda2`: Wavenumber at `I2`.
- `current_range`: Minimum and maximum operating current.

# Returns
- A two-element vector `[lambda_start, lambda_end]`.
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
    generate_intensity(instrument, I0, i0, i2, psi1, psi2, t)

Generate laser intensity over time from linear and second-harmonic intensity modulation terms.

# Arguments
- `instrument`: Instrument containing `mod_freq`.
- `I0`: Mean laser intensity.
- `i0`: Linear intensity-modulation coefficient.
- `i2`: Second-harmonic intensity-modulation coefficient.
- `psi1`: Phase offset of the linear term (rad).
- `psi2`: Phase offset of the second-harmonic term (rad).
- `t`: Time vector.

# Returns
- `light_intensity`: Intensity trace sampled at `t`.
"""
function generate_intensity(instrument, I0, i0, i2, psi1, psi2, t)
    # Calculate the intensity over time
    ω = 2π * instrument.mod_freq
    light_intensity = I0 * (1 .+ i0 * cos.(ω * t .+ psi1) .+ i2 * cos.(2 * ω * t .+ psi2))

    return light_intensity
end


"""
    design_filter(sampling_rate, cutoff_freq, filter_order, filter_type)

Construct a Butterworth low-pass or band-pass digital filter.

The returned object is intended for use with `filt(filter, signal)`.

# Arguments
- `sampling_rate::Real`: Sampling rate in Hz.
- `cutoff_freq::Union{Real, Tuple{Real, Real}}`: Cutoff frequency (low-pass)
  or `(f_low, f_high)` tuple (band-pass), in Hz.
- `filter_order::Int`: Butterworth filter order.
- `filter_type::String`: Either `"lowpass"` or `"bandpass"`.

# Returns
- A `digitalfilter` object.
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
    simulate_lockin(direct_signal, instrument, gain, harmonic)

Demodulate a signal with a sine reference at the selected harmonic.

# Arguments
- `direct_signal::AbstractArray`: Input time-domain signal.
- `instrument::Instrument`: Instrument configuration.
- `gain::Real`: Multiplicative lock-in gain.
- `harmonic::Int`: Harmonic index of the reference.

# Returns
- `lockin_y`: In-phase-to-sine demodulated component.
"""
function simulate_lockin(direct_signal::AbstractArray,
    instrument::Instrument,
    gain::Real,
    harmonic::Int)

    # Time axis aligned with the provided signal samples.
    t = range(0, stop=instrument.slow_sweep_period, length = length(direct_signal))
    # Multiply by the lock-in reference at the requested harmonic.
     lockin_y = direct_signal .* gain .* sin.(2π * harmonic * instrument.mod_freq .* t)
     # This may be useful later for phase-agnostic retrievals.
    # lockin_x = direct_signal .* gain .* cos.(2π * harmonic * instrument.mod_freq .* t)
    # Optionally combine quadratures as magnitude.
    # lockin_signal = sqrt.(lockin_x.^2 + lockin_y.^2)

    return lockin_y
end


"""
    sim_open_path_instrument(x, instrument, spectra, p, T, I0, i0, i2, psi1, psi2, t)

Run the forward instrument simulation for a methane open-path measurement.

Pipeline:
1) Compute transmission from spectroscopy and pathlength.
2) Interpolate transmission onto the instrument sampling grid.
3) Apply laser intensity modulation.
4) Band-pass filter around 2f and lock-in demodulate.
5) Low-pass filter the demodulated trace.

# Returns
- `lockin_2f_filtered`: Simulated 2f-like retrieval signal.
"""
function sim_open_path_instrument(x, instrument, spectra, p, T, I0, i0, i2, psi1, psi2, t)
    # Build the instrument grid and compute transmission on the spectroscopy grid.
    ν_grid = spectra["CH4"].grid
    instrument_grid = generate_spectral_grid(instrument)
    
    # Calculate transmission
    transmitance = SpectralFits.calculate_transmission(x, spectra,
    instrument.pathlength, length(ν_grid),
    p=p, T=T)
    
    # Map transmission from spectroscopy grid to instrument grid.
    transmitance = SpectralFits.apply_instrument(ν_grid, transmitance, instrument_grid)
    
    # Scale laser power by transmission to produce detector direct signal.
    light_intensity = generate_intensity(instrument, I0, i0, i2, psi1, psi2, t)[1:length(transmitance)]
    direct_signal = light_intensity .* transmitance
    
    # Isolate the 2f band, then demodulate with lock-in reference.
    # Band-pass keeps a +/-20% window around 2f.
    bandpass_filter = design_filter(instrument.sampling_rate, (1.8 * instrument.mod_freq, 2.2 * instrument.mod_freq), 6, "bandpass")
    signal_2f = filt(bandpass_filter, direct_signal)
    
    # Create the 2-F signal from the lock-in amplifier
    harmonic = 2
    gain = 1.0
    lockin_2f = simulate_lockin(signal_2f, instrument, gain, harmonic)
    
    # Low-pass smooths residual high-frequency structure in the demodulated output.
    lowpass_filter = design_filter(instrument.sampling_rate, 100.0, 5, "lowpass")
    lockin_2f_filtered = filt(lowpass_filter, lockin_2f)
    
    return lockin_2f_filtered
end