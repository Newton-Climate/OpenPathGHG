### this script is for running the retrieval

# import the instrument modeling functions
include("instrument_model.jl")
include("read_data.jl")
using Dates, CSV, DataFrames, Revise

# define the inversion hyper parameters
inversion_setup = Dict{String,Any}(
    "poly_degree" => 2, # degree of polynomial to fit the baseline
    "fit_pressure" => true,
    "fit_temperature" => true,
    "linear" => false,
        "verbose_mode" => true,
    "fit_column" => true, # fit for column density or concentration? (i.e., molec/cm² or ppb)
    "gamma" => 10.0, # regularization param where larger is a smaller step-size in the algorithm
    )

## define the atmospheric measurement conditions
# pressure (HPa) and temperature (K)
p, T = 1.0e3, 295.0
H2O = 0.0 # water vapor (humidity) in molecules/cm²


### read spectroscopy data
# Directory and data setup
datadir = "./line-lists/"
# define wave-number grid in cm for spectral model
ν_grid = 6040:0.005:6050
CH₄ = get_molecule_info("CH4", joinpath(datadir, "hit08_12CH4.par"), 6, 1, ν_grid)
# CH₄_C13 = get_molecule_info("13CH4", joinpath(datadir, "hit16_13CH4.par"), 6, 2, ν_grid)
# H2O = get_molecule_info("H2O", joinpath(datadir, "hit20_H2O.par"), 1, 1, ν_grid)
spectra = setup_molecules([CH₄])



# read the data
file_path = "../data/raw/overnight_0430/10mVsens_1Vmod_9kmod_0_0.txt"

# build the measurement
# extract experimental observations
# this is to calculate the noise in the measurement
#σ = calc_noise(measurement.intensity, plot_grid, 6046.5, 6047.75, poly_degree=2)
wavenumber_range = (6046.225205564099, 6047.815846426603)
measurement = read_measurement(file_path, wavenumber_range=wavenumber_range, p=p, T=T)

dynamic_range = abs.(maximum(measurement.intensity) - minimum(measurement.intensity))

# define the instrument parameters
# this is used in the forward model
include("build_instrument.jl")


# calculate the column density (num molecules per cm^2)
# in the leight-path
# define the column density of dry-air in the light-path (molecules/cm²)
vcd = SpectralFits.calc_vcd(p, T, instrument.pathlength, H2O)

## define the initial guess 
# units are in ppb * column density. this results in molecules/cm²
x = StateVector(
                  "CH4" => 1900e-9 * vcd,
                  "pressure" => p,
                  "temperature" => T,
                  "shift" => 0.0, # shift on the wave-number axis
                  "dc_offset" => 0.0, # shift in the direct current axis
                  "shape_parameters" => [dynamic_range; 1.0] # polynomial to fit baseline
)


# define a aprior uncertainty vector
# same units as x
# reps 1-sigma uncertainty
σ² = StateVector(
    "CH4" => 0.25 * x["CH4"],
    "pressure" => 0.1 * x["pressure"],
    "temperature" => 0.1 * x["temperature"],
    "shift" => 0.01, # cm⁻¹
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
f = sim_open_path_instrument(x, instrument, spectra, inversion_setup)
y = f(x)

### run the retrieval algorithm
out = adaptive_inversion(f, x, measurement, spectra, inversion_setup)

# recalculate vcd from newly fitted p and T
vcd = SpectralFits.calc_vcd.(out.x["pressure"], out.x["temperature"], instrument.pathlength)
CH4 = out.x["CH4"] / vcd
@show CH4
@show out.x

plot_grid = range(min_wavenumber, stop=max_wavenumber, length=length(out.model))
# plot the model and measurement and residual
p_compare = plot(plot_grid, out.measurement, label="measurement")
p_model = plot(plot_grid, out.model, label="model")
p_res= plot!(plot_grid, out.model - out.measurement, label="residual")
xlabel!("wave-number")
ylabel!("intensity")
plot(p_compare, p_model, p_res, layout=(3,1), legend=:bottomright)
savefig("inversion_result_adaptive_bayes.png")

# show the model - measurement percent error
percent_error = mean(100 * (out.model - out.measurement) ./ out.measurement)
@show percent_error