### End-to-end demonstration script for the open-path instrument model.
#
# This script:
# 1) Configures spectroscopy and instrument parameters.
# 2) Runs the forward instrument simulation.
# 3) Plots time-domain and frequency-domain diagnostics.

# Load local instrument-model utilities.
include("instrument.jl")
# Load dependencies for spectroscopy and plotting.
using SpectralFits, vSmartMOM.Absorption
using Plots, FFTW

### Spectroscopy setup
# Directory and line-list data setup.
datadir = "../../FreqComb/DCS_experiments/spectra/"
ν_grid = 6040:0.005:6050
CH₄ = get_molecule_info("CH4", joinpath(datadir, "hit08_12CH4.par"), 6, 1, ν_grid)
CH₄_C13 = get_molecule_info("13CH4", joinpath(datadir, "hit16_13CH4.par"), 6, 2, ν_grid)
H2O = get_molecule_info("H2O", joinpath(datadir, "hit20_H2O.par"), 1, 1, ν_grid)
spectra = setup_molecules([CH₄, H2O, CH₄_C13])

## Atmospheric measurement conditions
# Pressure (hPa) and temperature (K).
p, T = 1e3, 290
# Laser pathlength in cm.
pathlength= 500e2 # path in cm
# Calculate column density (molecules cm^-2) over the light path.
vcd = SpectralFits.calc_vcd(p, T, pathlength)

# Define the gas abundances.
x = Dict("CH4" => 2000e-9 * vcd,
"H2O" => 0.0 * vcd,
"13CH4" => 2000e-9 * vcd)

### read the dataset
# Ignore for now until retrieval
#laser_data = "data/lab_data_02-23/laser/"
#df = CSV.read(laser_data*"/ch4_meas_0_1.txt", DataFrame, delim=',', header=["cell", "room", "lockin"])
# filter out the rising parts of the wavelength sweep
#(lockin_rise, environmental_rise) = find_peaks(df)
# normalize the lockin-in signal to the environmental signal
# normalized_lockin = normalize_lockin_signal(environmental_rise, lockin_rise)

### Instrument parameterization
# Laser current range in mA.
I_start, I_end = 30.1, 45.1
# Reference current and wavelength points from lab characterization.
I1, λ1 = 36.6, 1653.678
I2, λ2 = 37.6, 1653.707
# Extrapolate instrument wavelength endpoints from the reference points.
λ_start, λ_end= calc_wavenumber_range(I1, I2, λ1, λ2, (I_start, I_end))
# Convert wavelength endpoints (nm) to wavenumber range (cm^-1).
# Note: start/stop swap is intentional.
min_wavenumber = 1e7 / λ_end
max_wavenumber = 1e7 / λ_start
wavenumber_range = (min_wavenumber, max_wavenumber)

## Laser modulation parameters
# Convert fast-modulation amplitude from nm to cm^-1.
mod_amplitude = 0.00367 # experimentally derived parameter for fast sweep 
mod_amplitude = convert_amplitude(min_wavenumber, max_wavenumber, mod_amplitude) 
mod_freq = 10.0e3 # fast sweep in hz
sampling_rate = 1.0e5 # samples / sec

# Vector length for one full sweep period.
num_samples = 2*250517
slow_sweep_period = num_samples*2/sampling_rate #sec per measurement sweep 
tuning_rate = (max_wavenumber - min_wavenumber) / slow_sweep_period
# Time vector for one full sweep period.
t = range(0, stop=slow_sweep_period, length=num_samples)

# Laser intensity-modulation parameters.
I0 = 4.0 # avg laser intensity
i0 = 0.00005 # amplitude of the linear IM
i2 = 0 # ignore the nonlinearities for the moment
psi1 = 0 # ignore phase shift rn
psi2 = 0 # ignore phase shift rn

# Build instrument configuration.
instrument = Instrument(wavenumber_range=wavenumber_range, mod_freq=mod_freq,
mod_amplitude=mod_amplitude, slow_sweep_period=slow_sweep_period,
num_samples=num_samples, pathlength=pathlength,
avg_laser_intensity=I0, tuning_rate=tuning_rate,
sampling_rate=sampling_rate
)

## Run the full simulation
model_test = sim_open_path_instrument(x, instrument, spectra, p, T, I0, i0,
    i2, psi1, psi2, t)

## Stepwise diagnostics
# Generate the spectral grid.
instrument_grid = generate_spectral_grid(instrument)
# Calculate transmission.
transmitance = SpectralFits.calculate_transmission(x, spectra,
    instrument.pathlength, length(ν_grid),
    p=p, T=T)
# Interpolate from spectroscopy grid to instrument grid.
transmitance = SpectralFits.apply_instrument(ν_grid, transmitance, instrument_grid)
# Scale laser power by transmission to get direct signal.
light_intensity = generate_intensity(instrument, I0, i0, i2, psi1, psi2, t)[1:length(instrument_grid)]
direct_signal = light_intensity .* transmitance

# Filter and demodulate using the lock-in workflow.
# Band-pass keeps a +/-20% window around 2f.
bandpass_filter = design_filter(sampling_rate, (1.8*instrument.mod_freq, 2.2*instrument.mod_freq), 6, "bandpass")
signal_2f = filt(bandpass_filter, direct_signal)
# Demodulate at the second harmonic.
harmonic = 2
gain = 1.0
lockin_2f = simulate_lockin(signal_2f, instrument, gain, harmonic)
# Smooth high-frequency residuals to highlight absorption line-shape.
lowpass_filter = design_filter(sampling_rate, 100.0, 5, "lowpass")
lockin_2f_filtered = filt(lowpass_filter, lockin_2f)

## Plot time-domain results
t = t[1:length(instrument_grid)]
p1 = plot(instrument_grid, signal_2f,
label="Filtered Direct Signal with 2-F Bandpass")
p2 = plot(instrument_grid, lockin_2f, label="Lock-in Signal")
ylabel!("Intensity")
p3 = plot(instrument_grid, lockin_2f_filtered,
label="Lock-in Signal Filtered by Low-pass")
xlabel!("wave-numbers (1/cm)")
plot(p1, p2, p3, layout=(3,1))
title!("Simulated signals")
savefig("time-domain-signal.png")

### Plot frequency-domain diagnostics
# Frequency axis for one-sided FFT.
f = fftfreq(length(t), sampling_rate)[1:length(t)÷2]

# Compute one-sided spectra.
fft_signal = fft(direct_signal)[1:length(t)÷2]
fft_signal_2f = fft(lockin_2f)[1:length(t)÷2]
fft_signal_2f_filtered = fft(lockin_2f_filtered)[1:length(t)÷2]

# Plot one-sided amplitude spectra.
p4 = plot(f, abs.(fft_signal), label="Direct Signal")
p5 = plot(f, abs.(fft_signal_2f), label="2-F Signal")
ylabel!("Intensity")
p6 = plot(f, abs.(fft_signal_2f_filtered), label="2-F Signal Filtered")
xlabel!("Frequency (Hz)")
plot(p4, p5, p6, layout=(3,1))
xlims!(8e3, 25e3)
savefig("frequency-domain-signal.png")

## Report power near 2f
tolerance = 10.0
target_freq = 2 * instrument.mod_freq

# Find the index of the closest frequency to 2*mod_freq
closest_index = argmin(abs.(f .- target_freq))

# Ensure the closest frequency is within the tolerance
if abs(f[closest_index] - target_freq) <= tolerance
    power_2f = abs2(fft_signal_2f[closest_index])
    power_2f_filtered = abs2(fft_signal_2f_filtered[closest_index])
    # do the same for the direct fft signal
    power_direct = abs2(fft_signal[closest_index])
    @show power_direct
    @show power_2f
    @show power_2f_filtered
else
    println("No frequency component found within the specified tolerance.")
end
