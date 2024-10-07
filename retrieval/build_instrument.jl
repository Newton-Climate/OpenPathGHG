### this script contains the instrument parameters 
# and the instrument object
# modify this script to define the insturment parameters 
# this is used in inversion and run_instrument_model scripts

# import an instrument function to build the wave-number grid
include("instrument_model.jl")
using DataFrames, CSV

# import the file as a dataframeV
# df = CSV.read(file_path, DataFrame, delim=',', header=["cell", "room", "lockin"])

### Define the instrument parameters
pathlength = 500.0e2 # cm
# laser current in miliamps
# max and min laser power in sweep
I_start, I_end = 30.1, 45.1
# reference current and wavelength from lab measurements
I1, λ1 = 36.6, 1653.678
I2, λ2 = 37.6, 1653.707
# extrapolate the wavenumber range of the instrument from lab reference-points
λ_start, λ_end= calc_wavenumber_range(I1, I2, λ1, λ2, (I_start, I_end))
# convert nanometers to wave-numbers (1/cm) of lambda_start:lambda_end
# Note: switch start and stop is intentional
min_wavenumber = 1e7 / λ_end
max_wavenumber = 1e7 / λ_start

# hard-code in from Cassandra
shift = 0.0
#min_wavenumber = 1e7 / 1653.83 + shift
#max_wavenumber = 1e7 / 1653.61 + shift

wavenumber_range = (min_wavenumber, max_wavenumber)
@show wavenumber_range

## define laser modulation parameters 
# convert amplitude of fast-modulation from nanometers to 1/cm
# experimentally derived parameter for fast sweep 
mod_amplitude = 0.00367 # nm
mod_amplitude = convert_amplitude(min_wavenumber, max_wavenumber, mod_amplitude) # cm⁻¹
mod_freq = 9.0e3 # fast sweep in hz
num_samples = 2*length(measurement.intensity)
slow_sweep_period = 5.0 #sec per measurement sweep 
sampling_rate = num_samples / slow_sweep_period # samples / sec

# length of the vector for one full period
tuning_rate = (max_wavenumber - min_wavenumber) / slow_sweep_period
# the time vector for one full sweep period
t = range(0, stop=slow_sweep_period, length=num_samples)

# parameters for modulation of laser light power
I0 = mean(measurement.room) # avg laser intensity
@show I0

## calculate the amplitude` of the fast-sweep modulation with the data
# Take a 9-k modulation frequency snippet
fast_sweep_period = 1.0 / mod_freq
# take the first 9-k modulation frequency snippet
fast_sweep = measurement.intensity[1:round(Int, fast_sweep_period * sampling_rate)]

# calculate the amplitude of the fast-sweep modulation
i0 = (maximum(fast_sweep) - minimum(fast_sweep)) / 2
i2 = 0.0 # ignore the nonlinearities for the moment
psi1 = 0.0 # ignore phase shift rn
psi2 = 0.0 # ignore phase shift rn

sensitivity = 0.01
gain = (10/sensitivity)*(pi/(2*sqrt(2)))
# build the laser power struct
laser_power = LaserPower(I0=I0, i0=i0, i2=i2, psi1=psi1, psi2=psi2, gain=gain)

# create the instrument object
instrument = Instrument(wavenumber_range=wavenumber_range, mod_freq=mod_freq,
mod_amplitude=mod_amplitude, slow_sweep_period=slow_sweep_period,
num_samples=num_samples, pathlength=pathlength,
avg_laser_intensity=I0, tuning_rate=tuning_rate,
sampling_rate=sampling_rate,
laser_power=laser_power
)
