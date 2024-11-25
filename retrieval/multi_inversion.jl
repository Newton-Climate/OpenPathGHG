### this script is for running the retrieval

# import the instrument modeling functions
using Dates, CSV, DataFrames, Revise, Plots

include("instrument_model.jl")
include("read_data.jl")
include("build_instrument.jl")


# define the inversion hyper parameters
inversion_setup = Dict{String,Any}(
    "poly_degree" => 2, # degree of polynomial to fit the baseline
    "fit_pressure" => false,
    "fit_temperature" => false,
    "linear" => false,
    "verbose_mode" => true,
    "fit_column" => true, # fit for column density or concentration? (i.e., molec/cm² or ppb)
    "gamma" => 17.0, # regularization param where larger is a smaller step-size in the algorithm
    )

## define the atmospheric measurement conditions
# pressure (HPa) and temperature (K)
# H2O = 2050e-1 # water vapor (humidity) in molecules/cm²


### read spectroscopy data
# Directory and data setup
datadir = "./retrieval/line-lists/"
# define wave-number grid in cm for spectral model

# read the data
file_path = "./data/raw/lab_test_1106/ch4_h2o_test_0.txt"

# build the measurement
# extract experimental observations
# this is to calculate the noise in the measurement
#σ = calc_noise(measurement.intensity, plot_grid, 6046.5, 6047.75, poly_degree=2)
wavenumber_range = (6046.225205564099, 6047.815846426603)

# start the for loop here
# for the number of iterations per measurement, run the inversion and add them to an array

#first define the measurement parameters and setup the names of the columns
num_meas_per_file = 200
num_channels_per_meas = 4
num_iter_meas = 1:200
num_cols = Int(num_channels_per_meas * num_meas_per_file)
col_numbers = 1:num_cols
col_names = ["Column$x" for x ∈ col_numbers]
i = Int(20)
CH4_array = []
H2O_array = []
p_array = []
T_array = []
percent_error_array = []
distance = 6766.0

p, T = 1013.25, 294.0

for k in num_iter_meas
    @show k
    H2O_guess = 5000e-6
    CH4_guess =  2020e-9

    ν_grid = 6040:0.005:6050
    CH₄ = get_molecule_info("CH4", joinpath(datadir, "hit08_12CH4.par"), 6, 1, ν_grid)
    # CH₄_C13 = get_molecule_info("13CH4", joinpath(datadir, "hit16_13CH4.par"), 6, 2, ν_grid)
    H2O = get_molecule_info("H2O", joinpath(datadir, "hit20_H2O.par"), 1, 1, ν_grid)
    spectra = setup_molecules([CH₄, H2O])

    df = CSV.read(file_path, DataFrame, delim=',', header=col_names)
    measurement = read_measurement(df, col_names, k, wavenumber_range=wavenumber_range, p=p, T=T)
    # dynamic_range = abs.(measurement.intensity[end] - measurement.intensity[1])/4
    delta = Int.(round(length(measurement.intensity)/4))
    dynamic_range = abs.(mean(measurement.intensity[end-delta:end]) - mean(measurement.intensity[1:1+delta]))/4
    offset = (mean(measurement.intensity[end-delta:end])+mean(measurement.intensity[1:1+delta]))/2

    # define the instrument parameters
    instrument, laser_power = build_instrument(measurement, distance)
    # this is used in the forward model

    # calculate the column density (num molecules per cm^2)
    # in the leight-path
    # define the column density of dry-air in the light-path (molecules/cm²)
    # vcd = SpectralFits.calc_vcd(p, T, instrument.pathlength, H2O)
    vcd = SpectralFits.calc_vcd(p, T, instrument.pathlength)
    @show vcd

    ## define the initial guess 
    # units are in ppb * column density. this results in molecules/cm²
    x = StateVector(
                    "CH4" => CH4_guess * vcd,
                    "H2O" => H2O_guess * vcd,
                    # "pressure" => p,
                    # "temperature" => T,
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
        # "pressure" => 0.01 * x["pressure"],
        # "temperature" => 0.01 * x["temperature"],
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
    f = sim_open_path_instrument(x, instrument, spectra, inversion_setup,measurement)

    y = f(x)

    #run measurement through simulated lockin
    harmonic = 2
    filter_window = 1000.0
    cutoff = 2000

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

    CH4 = (out.x["CH4"] / vcd)/10^-9
    H2O = (out.x["H2O"] / vcd)/10^-6
    percent_error = mean(100 * (out.model - out.measurement) ./ out.measurement)

    push!(CH4_array, CH4)
    push!(H2O_array, H2O)
    # push!(p_array, out.x["pressure"])
    # push!(T_array, out.x["temperature"])
    push!(percent_error_array, percent_error)

    # show the model - measurement percent error
    

    # global p = (out.x["pressure"] + p) / 2
    # global T = (out.x["temperature"] + T) / 2
    
    # global CH4_guess = ((out.x["CH4"] / vcd) + CH4_guess) / 2
    # global H2O_guess = ((out.x["H2O"] / vcd) + CH4_guess) / 2

end


# plot the model and measurement and residual
p_ch4 = plot(num_iter_meas, CH4_array, label="methane")
p_h2o = plot(num_iter_meas, H2O_array, label="water")
# p_t = plot(num_iter_meas, p_array, label="pressure")
# p_p = plot(num_iter_meas, T_array, label="temperature")
p_error = plot(num_iter_meas, percent_error_array, label = "error")

plot(p_ch4,p_h2o,p_error, layout=(3,1), legend=:bottomright)
savefig("mutli_estimation.png")