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
using DSP, Waveforms, Polynomials, StatsBase, FFTW
using Interpolations

"""
struct LaserPower{FT}
    Struct that stores the laser power modulation parameters
    ================================
    # Fields
    - `I0`: (mW/m²) The average laser intensity
    - `i0`: (mW/m²) The amplitude of the linear intensity modulation (IM)
    - `i2`: (mW/m²) The amplitude of the nonlinear intensity modulation 
    - `psi1`: (rad) The phase shift for linear IM/FM (ignored for now)
    - `psi2`: (rad) The phase shift for nonlinear IM/FM (ignored for now)
    - `gain`: (float) The gain of the lock-in amplifier
    """
Base.@kwdef struct LaserPower{FT}
    I0::FT
    i0::FT
    i2::FT
    psi1::FT
    psi2::FT
    min_room::FT
end


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
sampling_rate::FT`:`: (Hz) The sampling rate of the instrument

===============================

# Example
instrument = Instrument(wavenumber_range=wavenumber_range, mod_freq=mod_freq,
mod_amplitude=mod_amplitude, slow_sweep_period=slow_sweep_period,
num_samples=num_samples, pathlength=pathlength,
avg_laser_intensity=I0, tuning_rate=tuning_rate, sampling_rate=sampling_rate
)
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
    laser_power::LaserPower{FT}
    gain::FT
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
    num_samples::Int,
    t:: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64})

    # Generate the time-vector, which is slow_sweep_period long
    # t is only half a cycle 
    # for sweep of laser
    # t = range(0, stop=slow_sweep_period, length=num_samples)
    # Generate the triangle wave using the trianglewave function
    slow_sweep_freq = 1 / slow_sweep_period
    triangle_wave = peak_value * abs.(sawtoothwave.(2π * slow_sweep_freq .* t))

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
    num_samples::Int,
    t:: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64};
    phase_shift::Real=0.0)

    # Generate the time-vector to cover num_cycles * 2π within half the period
    # t = range(0, stop=slow_sweep_period, length=num_samples)
    # Generate the fast sweep using sine function with phase shift
    fast_sweep = mod_amplitude * sin.(2π * mod_freq .* t .+ phase_shift)
    
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
function generate_spectral_grid(instrument::Instrument, t:: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64})
    # Generate the slow sweep with triangle wave
    # amplitude is the max minus min wave-number
    peak_value = instrument.wavenumber_range[2] - instrument.wavenumber_range[1]
    slow_sweep = generate_triangle_wave(peak_value, instrument.slow_sweep_period, instrument.num_samples,t)

    # Generate the fast sweep
    fast_sweep = generate_fast_sweep(instrument.mod_amplitude, instrument.mod_freq, instrument.slow_sweep_period, instrument.num_samples,t)

    # Generate the wave-number grid by adding the 
    # slow and fast sweeps 
    # and starting at min_wavenumber
    starting_wavenumber = minimum(instrument.wavenumber_range)
    wavenumber_grid = slow_sweep .+ fast_sweep .+ starting_wavenumber

    return wavenumber_grid
end


"""
calculate the max and min wave-length based on the laboratory reference points in the lab data.
Reference points are the current (in miliamps)values at which the wavenumber is known.
================================================================
# Parameters:
- `I1`: Current at the first reference point
- `I2`: Current at the second reference point
- `lambda1`: Wavelength at the first reference point
- `lambda2`: Wavelength at the second reference point
- `current_range`: Tuple of the min and max current values
===============================
# Returns:
- `wavelength_range`: Tuple of the min and max wavelengths
"""
function calc_wavelength_range(t0, t1, λ_center, t_center)
    
    m = 0.0912/0.013639 #nm/s
    # m = 0.0912/7.5295 #nm/mA 

    λ0 = m*(t0-t_center) + λ_center
    λ1 = m*(t1-t_center) + λ_center

    # m = 0.0909/8.178 #nm/mA 
    # 89.48
    # lambda_start = m*(I1-83.99)+1653.738
    # lambda_end = m*(I2-83.99)+1653.738
    # lambda_start = m*(I1-83.60)+1653.728
    # lambda_end = m*(I2-83.60)+1653.728
    # Return the max and min wavelength 
    return [λ0, λ1]
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
function generate_intensity(instrument, I0, i0, i2, psi1, psi2, min_room, t)
    # Calculate the intensity over time
    ω = 2π * instrument.mod_freq
    @show I0
    @show i0
    loss = 1
    triangle_wave_IM = I0 * abs.(sawtoothwave.(2π * (1/instrument.slow_sweep_period) .* t)) .+ min_room .+i0
    light_intensity = (triangle_wave_IM  .+ i0 * sin.(ω * t .+ psi1) .+ i2 * sin.(2 * ω * t .+ psi2))/loss

    return light_intensity
end


function generate_intensity(instrument, t)
    laser = instrument.laser_power
    return generate_intensity(instrument, laser.I0, laser.i0, laser.i2, laser.psi1, laser.psi2, laser.min_room, t)
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


function apply_filter(signal, filter_obj, mod_freq, mod_period, fs, padding_type="reflective")
    # Calculate padding length: pad enough to cover slow modulation period
    pad_length = 1*round(Int, fs * mod_period)

    # # Apply padding based on the selected type
    # if padding_type == "zero"
    #     padded_signal = vcat(zeros(pad_length), signal, zeros(pad_length))
    # elseif padding_type == "reflective"
    #     padded_signal = vcat(reverse(signal[1:pad_length]), signal, reverse(signal[end-pad_length+1:end]))
    # elseif padding_type == "replicative"
    #     padded_signal = vcat(signal[1:pad_length], signal, signal[end-pad_length+1:end])
    # else
    #     error("Unsupported padding type: $padding_type")
    # end

    # Filter the signal
    filtered_signal = filtfilt(filter_obj, signal)
    # Remove padding to restore original signal length
    # filtered_signal = filtered_signal[pad_length+1:end-pad_length]

    return filtered_signal
end


"""
simulate the lock-in amplifier

    ================================

    Parameters:
    - `direct_signal`: the direct signal
    - `instrument`: The instrument object
    - `gain`: The gain of the lock-in amplifier
    - `harmonic`: The harmonic of the lock-in amplifier
    - 't': time of the full period sweep
    ================================

    Returns:
    - `lockin_signal`: The lock-in signal
"""

function simulate_lockin(direct_signal,instrument,gain,harmonic,t,lockin_component)

    # create the time vector
    # Generate the 2F signal
    lockin_y = direct_signal .* gain .* sin.(2π * harmonic * instrument.mod_freq .* t)
     # this may be be needed later for phase-agnostic retrievals
    lockin_x = direct_signal .* gain .* cos.(2π * harmonic * instrument.mod_freq .* t)
    # Multiply the direct signal by the 2F signal
    lockin_signal = sqrt.(lockin_x.^2 + lockin_y.^2)

    # return lockin_signal, lockin_x, or lockin_y depending on a flat_region
    if lockin_component == "x"
        return reverse(lockin_x)
    elseif lockin_component == "y"
        return reverse(lockin_y)
    elseif lockin_component == "squared"
        return lockin_signal
    else
        # raise an error
        error("lockin_component must be one of 'x', 'y', or 'squared'")
    end
end


function test_apply_instrument(og_grid, spectra, new_grid)

    new_spec = []
    itp = LinearInterpolation(og_grid, spectra)    
    for i in new_grid
        new_spec = push!(new_spec,itp(i))
    end
    # new_spec = itp(new_grid)

    return new_spec

end

function run_bandpass_filter(signal, filter_window, modulation_freq, harmonic, sampling_rate, slow_sweep_period)
    
    window_left = harmonic * modulation_freq - filter_window
    window_right = harmonic * modulation_freq + filter_window
    bandpass_filter = design_filter(sampling_rate, (window_left, window_right), 1, "bandpass")
    bp_filt_signal = apply_filter(signal, bandpass_filter, modulation_freq, slow_sweep_period, sampling_rate)
    return bp_filt_signal

end

function run_lowpass_filter(signal, cutoff, modulation_freq, sampling_rate, slow_sweep_period)
    
    lowpass_filter = design_filter(sampling_rate, cutoff, 1, "lowpass")   
    lockin_2f_filtered = apply_filter(signal, lowpass_filter, modulation_freq, slow_sweep_period, sampling_rate)

    return lockin_2f_filtered
end

function sim_open_path_instrument(x0, instrument, spectra, inversion_setup, measurement)
    x_fields = collect(keys(x0))

    function f(x)
        if typeof(x) <: AbstractArray
            x = assemble_state_vector!(x, x_fields, inversion_setup)
        end
        
        # Generate the spectral grid
        ν_grid = spectra["CH4"].grid
        t = range(0, stop=instrument.slow_sweep_period, length=instrument.num_samples)
        instrument_grid = generate_spectral_grid(instrument,t)

        # check if we are fitting for pressure and temperature
        p = haskey(x, "pressure") ? x["pressure"] : measurement.pressure
        T = haskey(x, "temperature") ? x["temperature"] : measurement.temperature
        shift = haskey(x, "shift") ? x["shift"] : 0.0
        
        # Calculate transmission
        transmitance = SpectralFits.calculate_transmission(x, spectra,
        instrument.pathlength, length(ν_grid),
        p=p, T=T, fit_column=inversion_setup["fit_column"])
        # Interpolate from the spectroscopy model to the instrument grid
        instrument_grid = instrument_grid .+ shift
        transmitance = SpectralFits.apply_instrument(ν_grid, transmitance, instrument_grid)
        
        # Scale the laser power by the transmitance to get direct signal
        light_intensity = generate_intensity(instrument, t)
        test_intensity = 
        βloss = 0.9882
        gain_room_pd = 4.75e3
        responsivity_room = 0.95

        # Note: add sintolation function for turbulence-induced power fluctuations here.

        # scale the transmitance by the laser power
        # direct_signal = gain_room_pd .* responsivity_room .* light_intensity .* (1- βloss) .* (reverse(transmitance))
        direct_signal = light_intensity .* (reverse(transmitance))

        # Filter the signal and simulate lock-in amplifier
    
        # define constants
        harmonic = 2
        filter_window = 1000.0
        cutoff = 2000
        #run simulation through bandpass filter
        
        bp_filt_signal = run_bandpass_filter(direct_signal, filter_window, instrument.mod_freq, harmonic, instrument.sampling_rate, instrument.slow_sweep_period)

        #filter for the measured signal process
        
        # Create the 2-F signal from the lock-in amplifier
        lockin_2f = simulate_lockin(bp_filt_signal, instrument, 1, harmonic, t, "x")
        
        # Lowpass filter out high-frequency variations to highlight absorption line-shape
        lockin_2f_filtered = instrument.gain .* run_lowpass_filter(lockin_2f, cutoff, instrument.mod_freq, instrument.sampling_rate, instrument.slow_sweep_period)
        
        # calculate and scale intensity by measurement baseline 
        if "shape_parameters" in x_fields
            lockin_2f_filtered = lockin_2f_filtered .+ calc_polynomial_term(inversion_setup["poly_degree"], x["shape_parameters"], length(instrument_grid))
            # lockin_2f_filtered = lockin_2f_filtered        
        end

        if "dc_offset" in x_fields
            lockin_2f_filtered = lockin_2f_filtered .+ x["dc_offset"]
        end

        # the portion of signal returned is half a period
        # this is to ensure that the rising part of the signal is returned
        # inds = 1:Int(div(length(t), 2))
        inds = 1:Int(div(length(t), 2))

        # room = plot(t[inds], og_meas, label = "measurement for comparison")
        # p_filt= plot(lp_inst_grid[inds], signal_2f[inds], label = "measurement for comparison")
        # p_direct = plot(t[inds], lockin_2f_filtered[inds], label = "simulation then simulated lockin")
        # plot!(t_half, lockin_2f_filtered_meas_x, label = "measured simulated lock x")
        # plot!(t_half, lockin_2f_filtered_meas_y, label = "measured simulated lock y")
        # plot!(t[inds], inst_grid_test[inds]/10 .- 646.5, label = "sine grid")

        # plot!(t[inds], test[inds], label = "regular 18k sine")

        # p_meas = plot!(t[inds], light_intensity[inds],label="light intesnity")
        # ylims!(-0.2,0.2)

        # p_filt = plot(t_test,lockin_2f_filtered2, label = "interpolated lockin")
        # plot!(t[inds],cos_examp[inds], label = "cosine example")
        # ylims!(-0.1,0.1)
        
        # xlims!(0.005,0.045)
        # ylims!(-0.01,0.01)
        # plot!(p_direct, layout=(1,1), legend=:topright)
        
        # savefig("sim_meas2.png")
        return lockin_2f_filtered[inds]
    end

    return f
end

function test_sim_open_path_instrument(x, instrument, spectra, inversion_setup, measurement)


    # Generate the spectral grid
    ν_grid = spectra["CH4"].grid
    t = range(0, stop=instrument.slow_sweep_period, length=instrument.num_samples)
    instrument_grid = generate_spectral_grid(instrument,t)
    inds = 1:Int(div(length(t), 2))

    # check if we are fitting for pressure and temperature
    p = haskey(x, "pressure") ? x["pressure"] : measurement.pressure
    T = haskey(x, "temperature") ? x["temperature"] : measurement.temperature
    shift = haskey(x, "shift") ? x["shift"] : 0.0

    # Calculate transmission
    transmitance = SpectralFits.calculate_transmission(x, spectra,
    instrument.pathlength, length(ν_grid),
    p=p, T=T, fit_column=inversion_setup["fit_column"])
    # Interpolate from the spectroscopy model to the instrument grid
    instrument_grid = instrument_grid .+ shift
    transmitance = SpectralFits.apply_instrument(ν_grid, transmitance, instrument_grid)

    # Scale the laser power by the transmitance to get direct signal
    light_intensity = generate_intensity(instrument, t)
    βloss = 0.9882
    gain_room_pd = 4.75e3
    responsivity_room = 0.95
    ω = 2π * instrument.mod_freq
    t_half = t[inds]
    # test_intense = fit  .+ (gain_room_pd .* responsivity_room .* laser_power.i0 .*(1-βloss)) * ((sin.(ω * t_half .+ laser_power.psi1) .+ laser_power.i2 * sin.(2 * ω * t_half .+ laser_power.psi2)))
    # Note: add sintolation function for turbulence-induced power fluctuations here.

    # scale the transmitance by the laser power
    direct_signal = light_intensity .* (reverse(transmitance))
    # direct_signal = gain_room_pd .* responsivity_room .* light_intensity .* (1- βloss) .* (reverse(transmitance))
    # direct_signal = (reverse(transmitance[inds])).* test_intense

    # Filter the signal and simulate lock-in amplifier

    # define constants
    harmonic = 2
    filter_window = 1000.0
    cutoff = 2000
    #run simulation through bandpass filter

    bp_filt_signal = run_bandpass_filter(direct_signal, filter_window, instrument.mod_freq, harmonic, instrument.sampling_rate, instrument.slow_sweep_period)

    #filter for the measured signal process

    # Create the 2-F signal from the lock-in amplifier
    lockin_2f = simulate_lockin(bp_filt_signal, instrument, 1, harmonic, t, "x")

    # Lowpass filter out high-frequency variations to highlight absorption line-shape
    lockin_2f_filtered = instrument.gain .* run_lowpass_filter(lockin_2f, cutoff, instrument.mod_freq, instrument.sampling_rate, instrument.slow_sweep_period)
    direct_filt = run_lowpass_filter(direct_signal, 1000, instrument.mod_freq, instrument.sampling_rate, instrument.slow_sweep_period)

    # calculate and scale intensity by measurement baseline 
    # if "shape_parameters" in x_fields
    #     lockin_2f_filtered = lockin_2f_filtered .+ calc_polynomial_term(inversion_setup["poly_degree"], x["shape_parameters"], length(instrument_grid))
    #     # lockin_2f_filtered = lockin_2f_filtered        
    # end

    # if "dc_offset" in x_fields
    #     lockin_2f_filtered = lockin_2f_filtered .+ x["dc_offset"]
    # end

    # the portion of signal returned is half a period
    # this is to ensure that the rising part of the signal is returned
    inds = 1:Int(div(length(t), 2))

    # room = plot(t[inds], og_meas, label = "measurement for comparison")
    # p_filt= plot(lp_inst_grid[inds], signal_2f[inds], label = "measurement for comparison")
    p_direct = plot(t[inds], lockin_2f_filtered[inds], label = "simulation then simulated lockin")
    # plot!(t_half, lockin_2f_filtered_meas_x, label = "measured simulated lock x")
    # plot!(t_half, lockin_2f_filtered_meas_y, label = "measured simulated lock y")
    # plot!(t[inds], inst_grid_test[inds]/10 .- 646.5, label = "sine grid")

    # plot!(t[inds], test[inds], label = "regular 18k sine")

    # p_meas = plot!(t[inds], light_intensity[inds],label="light intesnity")
    # ylims!(-0.2,0.2)

    # p_filt = plot(t_test,lockin_2f_filtered2, label = "interpolated lockin")
    # plot!(t[inds],cos_examp[inds], label = "cosine example")
    # ylims!(-0.1,0.1)

    # xlims!(0.005,0.045)
    # ylims!(-0.01,0.01)
    plot(p_direct, layout=(1,1), legend=:bottomright)

    savefig("sim_meas2.png")
    return lockin_2f_filtered[inds]
    end




"""
calculate the noise in the measurement
Logic:
- Find the spectral region where the signal is flat
De-trend the baseline trend with a fitted polynomial
- Calculate the standard deviation of the residual
- Fit a noise model that is linear with offset term
- Calculate the noise from the model
- Return the noise
===============================
Parameters:
- `measurement_intensity`::Vector: The intensity of the measurement
- `measurement_grid`::Vector: The wavenumber grid of the measurement
- `ν_min`::Float: The minimum wavenumber
- `ν_max`::Float: The maximum wavenumber
- `poly_degree`::Int: The degree of the polynomial to fit
===============================
Returns:
- `σ`::Float: The noise in the measurement
"""
    function calc_noise(measurement_intensity::Vector{FT},
    measurement_grid::Vector{FT},
    ν_min::FT,
    ν_max::FT;
    poly_degree::Int=2) where FT <: AbstractFloat

    # select spectrally flat region 
    # find wavelength indices:
    ind = intersect(findall(measurement_grid .> ν_min), findall(measurement_grid .< ν_max))
    signal_total = mean(abs.(measurement_intensity))

    # Subset data for spectral range
    flat_region = measurement_intensity[ind]
    flat_grid = measurement_grid[ind] .- mean(measurement_grid[ind])

    # Fit a polynomial to the flat region to de-trend baseline
    fitted = Polynomials.fit(flat_grid, flat_region, poly_degree)
    println("done with polynomial fit")
    mod = fitted.(flat_grid)
    println("done with polynomial evaluation")
    # Mean residual (has little impact here)
    mm = mean(mod .- flat_region)
    # Standard deviation from fit (mean residual removed):
    #show the sizes of the arrays
    @show size(mod)
    @show size(flat_region)
    @show size(mm)
    sm = std(mod .- flat_region .- mm)

    # Fit noise model (linear with offset):
    @show size(signal_total)
    @show size(sm)
    slope_noise_fit = sqrt.(signal_total) \ sm
    println("fitting noise model")

    # This will now give you the total noise, 
    # i.e. for an individual (single) sounding,
    # the noise 1sigma is just 
    # Se = slope_noise_fit * sqrt(signal_total[:,1])
    σ = mean(slope_noise_fit * sqrt.(signal_total))

    return σ
end


# Function to manually compute second derivatives using the entire grid for each calculation
function calc_second_derivative(x, spectra, pathlength; 
    p=1e3, T=290, fit_column=false,
    use_OCO_table=false, adjust_ch4_broadening=false)

n = length(spectra["CH4"].grid)
FT = SpectralFits.dicttype(x)
# define a second_deriv array with type FT

second_derivatives = Array{FT}(undef, n)
grid = zeros(n)

# Call absorption cross-section for the entire grid each time to conform with expected input types
h = 0.001
σ = SpectralFits.calculate_transmission(x, spectra, pathlength, n,
    p=p, T=T, fit_column=fit_column,
    use_OCO_table=use_OCO_table, adjust_ch4_broadening=adjust_ch4_broadening)

for i in 2:n-1
grid[i] = spectra["CH4"].grid[i]
second_derivatives[i] = (σ[i+1] - 2 * σ[i] + σ[i-1]) ./ h^2
end

grid = grid[2:end-1]
second_derivatives = second_derivatives[2:end-1]

return second_derivatives, grid
end


"""
This is a function thaht generates a forward model function that 
    directly simulates 
    the second derivative of transmitance.
    This is useful for the inversion of the instrument measurements 
        and is a simpler mathematical representation of the lockin-in measurement
        ================================
        Parameters:
        - `state_vector`: (Dict or Vector)The state vector contaiing the parameters to retrieve
        - `instrument`: The instrument object
        - `spectra`: (Dict) the spectroscopy model parameters
        - `I0`: (Float) The average laser intensity
        - `i0`: (Float) The amplitude of the linear intensity modulation (IM)
        - `i2`: (Float) The amplitude of the nonlinear intensity modulation (IM) (ignored for now)
        - `psi1`: (Float) The phase shift for linear IM/FM (ignored for now)
        - `psi2`: (Float) The phase shift for nonlinear IM/FM (ignored for now)
        - `gain`: (Float) The gain of the lock-in amplifier
        - `inversion_setup`: (Dict) The inversion setup parameters
        ================================
        Returns:
        - `f`: The forward model function
"""
function second_deriv_model(state_vector, instrument, spectra, I0, i0, i2, psi1, psi2, gain, inversion_setup)
    x_fields = collect(keys(state_vector))

    function f(x)
        if typeof(x) <: AbstractArray
            x = assemble_state_vector!(x, x_fields, inversion_setup)
        end
        
        # assign the spectral grid from Hitran
        # ν_grid = spectra["CH4"].grid
        t = range(0, stop=instrument.slow_sweep_period, length=instrument.num_samples)

        # check if we are fitting for pressure and temperature
        p = haskey(x, "pressure") ? x["pressure"] : measurement.pressure
        T = haskey(x, "temperature") ? x["temperature"] : measurement.temperature
        shift = haskey(x, "shift") ? x["shift"] : 0.0
        
        # Calculate transmission
        model_transmitance, model_grid = calc_second_derivative(
            x, spectra, instrument.pathlength, p=p, T=T, fit_column=inversion_setup["fit_column"])
            # convert model_grid to an AbstractRange
            model_grid = range(model_grid[1], stop=model_grid[end], length=length(model_grid))

        
        # Interpolate from the spectroscopy model to the instrument 
        (wn_min, wn_max) = instrument.wavenumber_range
        instrument_grid = collect(range(wn_min, stop=wn_max, length=instrument.num_samples))
    
        instrument_grid = instrument_grid .+ shift
        instrument_transmitance = SpectralFits.apply_instrument(
            model_grid, model_transmitance,
            instrument_grid)
            light_intensity = generate_intensity(instrument, I0, i0, i2, psi1, psi2, t)
            instrument_intensity = light_intensity .* instrument_transmitance
            harmonic = 2
            lockin_2f = simulate_lockin(instrument_transmitance, instrument, gain, harmonic)
        
        # Filter out high-frequency variations to highlight absorption line-shape
        lowpass_filter = design_filter(instrument.sampling_rate, 100.0, 2, "lowpass")
        
        lockin_2f_filtered = apply_filter(lockin_2f, lowpass_filter, instrument.mod_freq, instrument.slow_sweep_period, instrument.sampling_rate)
        
        # calculate and scale intensity by measurement baseline
        if "shape_parameters" in x_fields
            lockin_2f_filtered = lockin_2f_filtered .+ calc_polynomial_term(inversion_setup["poly_degree"], x["shape_parameters"], length(instrument_grid))
        end

        if "dc_offset" in x_fields
            lockin_2f_filtered .+= x["dc_offset"]
        end
        
        # the portion of signal returned is half a period
        # this is to ensure that the rising part of the signal is returned
        inds = 1:Int(div(length(lockin_2f_filtered), 2))
        
        return lockin_2f_filtered[inds]
    end

    return f
end