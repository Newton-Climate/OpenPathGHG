### this script is for running the retrieval

# import the instrument modeling functions
using Dates, CSV, DataFrames, Revise, Plots, CurveFit

include("instrument_model.jl")
include("read_data.jl")
include("build_instrument.jl")


# define the inversion hyper parameters
inversion_setup = Dict{String,Any}(
    "poly_degree" => 2, # degree of polynomial to fit the baseline
    "fit_pressure" => true,
    "fit_temperature" => true,
    "linear" => false,
    "verbose_mode" => true,
    "fit_column" => true, # fit for column density or concentration? (i.e., molec/cm² or ppb)
    "gamma" => 17.0, # regularization param where larger is a smaller step-size in the algorithm
    )

## define the atmospheric measurement conditions
# pressure (HPa) and temperature (K)
p, T = 1013.25, 294.0
# H2O = 2050e-1 # water vapor (humidity) in molecules/cm²
distance = 6766.0

### read spectroscopy data
# Directory and data setup
datadir = "./retrieval/line-lists/"
# define wave-number grid in cm for spectral model
ν_grid = 6040:0.005:6050
CH₄ = get_molecule_info("CH4", joinpath(datadir, "hit08_12CH4.par"), 6, 1, ν_grid)
# CH₄_C13 = get_molecule_info("13CH4", joinpath(datadir, "hit16_13CH4.par"), 6, 2, ν_grid)
H2O = get_molecule_info("H2O", joinpath(datadir, "hit20_H2O.par"), 1, 1, ν_grid)
spectra = setup_molecules([CH₄, H2O])



# read the data
file_path = "./data/raw/1122_run_through.txt"

# build the measurement
# extract experimental observations
# this is to calculate the noise in the measurement
#σ = calc_noise(measurement.intensity, plot_grid, 6046.5, 6047.75, poly_degree=2)
wavenumber_range = (6046.225205564099, 6047.815846426603)
num_meas_per_file = 200
num_channels_per_meas = 4
num_cols = Int(num_channels_per_meas * num_meas_per_file)
col_numbers = 1:num_cols
col_names = ["Column$x" for x ∈ col_numbers]

k=1

# room_direct_filtered = run_lowpass_filter(measurement.room, 1000, instrument.mod_freq, sampling_rate_half, instrument.slow_sweep_period/2)

# fit = CurveFit.curve_fit(Polynomial, t_half, room_direct_filtered, 2)

df = CSV.read(file_path, DataFrame, delim=',', header=col_names)

fit = CurveFit.curve_fit(Polynomial,1:length(df[!,1]), df[!,1], 20)
y_fit = fit.(1:length(df[!,1]))

test_fit = plot(1:length(df[!,1]),df[!,1])
plot!(1:length(df[!,1]),y_fit)
plot(test_fit, layout=(1,1), legend=:bottomright)
savefig("test_fit.png")
df[!,1] = df[!,2]

# measurement = average_measurements(df, col_names, num_iter_meas, wavenumber_range, p, T)
measurement = read_measurement(df, col_names, k, wavenumber_range=wavenumber_range, p=p, T=T)
# dynamic_range = abs.(measurement.intensity[end] - measurement.intensity[1])/4
delta = Int.(round(length(measurement.intensity)/4))
dynamic_range = abs.(mean(measurement.intensity[end-delta:end]) - mean(measurement.intensity[1:1+delta]))/4
offset = (mean(measurement.intensity[end-delta:end])+mean(measurement.intensity[1:1+delta]))/2
@show argmin(measurement.intensity)
@show length(measurement.intensity)
@show dynamic_range
p_room_meas = plot(measurement.grid, reverse(measurement.intensity), label="room")
p_lock_meas = plot(measurement.grid, measurement.room, label="lockin")
p_cell_meas = plot(collect(range(wavenumber_range[1], wavenumber_range[2], length=length(measurement.cell))), measurement.cell, label="lockin")
plot(p_room_meas, p_lock_meas, p_cell_meas, layout=(3,1), legend=:bottomright)
savefig("room_meas.png")

# define the instrument parameters
# this is used in the forward model
instrument, laser_power = build_instrument(measurement, distance)

H2O = 5000e-6
# calculate the column density (num molecules per cm^2)
# in the leight-path
# define the column density of dry-air in the light-path (molecules/cm²)
# vcd = SpectralFits.calc_vcd(p, T, instrument.pathlength, H2O)
vcd = SpectralFits.calc_vcd(p, T, instrument.pathlength)
@show vcd

## define the initial guess 
# units are in ppb * column density. this results in molecules/cm²
x = StateVector(
                  "CH4" => 2020e-9 * vcd,
                  "H2O" => 5000e-6 *vcd,
                  "pressure" => p,
                  "temperature" => T,
                  "shift" => 0.0, # shift on the wave-number axis
                  "dc_offset" => offset, # shift ins the direct current axis
                  "shape_parameters" => [dynamic_range; 0.005] # polynomial to fit baseline
)


# define a aprior uncertainty vector
# same units as x
# reps 1-sigma uncertainty
σ² = StateVector(
    "CH4" => 0.35 * x["CH4"],
    "H2O" => 0.35 * x["H2O"],
    "pressure" => 0.01 * x["pressure"],
    "temperature" => 0.01 * x["temperature"],
    "shift" => 0.001, # cm⁻¹
    "dc_offset" => 0.01 * dynamic_range,
    "shape_parameters" => 0.1 .* ones(inversion_setup["poly_degree"])
)
# add prior uncertainty to the inversion_setup as an inversion parameter
inversion_setup["σ"] = σ²

### define the instrument and forward model
# this is the simplified model
# Note that this is being tested
#f = second_deriv_model(x, instrument, spectra, I0, i0, i2, psi1, psi2, gain, inversion_setup)

# this is the full model
#f = sim_open_path_instrument(x, instrument, spectra, I0, i0, i2, psi1, psi2, gain, inversion_setup)
# f = sim_open_path_instrument(x, instrument, spectra, inversion_setup)
f = sim_open_path_instrument(x, instrument, spectra, inversion_setup, measurement)

y = f(x)

#run measurement through simulated lockin
harmonic = 2
filter_window = 1000.0
cutoff = 1000

t_half = range(0, stop=instrument.slow_sweep_period/2, length=length(measurement.room))
sampling_rate_half = length(measurement.room)/t_half[end]
bp_filt_meas = run_bandpass_filter(measurement.room, filter_window, instrument.mod_freq, harmonic, sampling_rate_half, instrument.slow_sweep_period/2)
        
meas_lock_x = simulate_lockin(bp_filt_meas, instrument, 1, harmonic, t_half, "x")
meas_lock_y = simulate_lockin(bp_filt_meas, instrument, 1, harmonic, t_half,"y")

lockin_2f_filtered_meas_x = instrument.gain .* run_lowpass_filter(meas_lock_x, cutoff, instrument.mod_freq, sampling_rate_half, instrument.slow_sweep_period/2)
lockin_2f_filtered_meas_y = instrument.gain .* run_lowpass_filter(meas_lock_y, cutoff, instrument.mod_freq, sampling_rate_half, instrument.slow_sweep_period/2)

### run the retrieval algorithm

measurement.intensity = lockin_2f_filtered_meas_x
out = adaptive_inversion(f, x, measurement, spectra, inversion_setup)

# recalculate vcd from newly fitted p and T
# vcd = SpectralFits.calc_vcd.(out.x["pressure"], out.x["temperature"], instrument.pathlength)
# vcd = SpectralFits.calc_vcd.(p, T, instrument.pathlength, (out.x["H2O"] / vcd))
vcd = SpectralFits.calc_vcd.(p, T, instrument.pathlength)

@show vcd
CH4 = (out.x["CH4"] / vcd)/10^-9
H2O = (out.x["H2O"] / vcd)/10^-6

@show CH4 
@show H2O
@show out.x

plot_grid = range(0, stop=0.05, length=length(out.model))
# plot the model and measurement and residual
p_compare = plot(plot_grid, out.measurement, label="measurement")
p_model = plot!(plot_grid, out.model, label="model")
p_res= plot!(plot_grid, out.model - out.measurement, label="residual")
xlabel!("wave-number")
ylabel!("intensity")
plot(p_compare, layout=(1,1), legend=:bottomright)
savefig("inversion_result_adaptive_bayes.png")

# show the model - measurement percent error
percent_error = mean(100 * (out.model - out.measurement) ./ out.measurement)
@show percent_error