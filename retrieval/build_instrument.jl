    ### this script contains the instrument parameters 
    # and the instrument object
    # modify this script to define the insturment parameters 
    # this is used in inversion and run_instrument_model scripts

    # import an instrument function to build the wave-number grid
    include("instrument_model.jl")
    using DataFrames, CSV
function build_instrument(measurement, distance)
    # import the file as a dataframeV
    # df = CSV.read(file_path, DataFrame, delim=',', header=["cell", "room", "lockin"])

    ### Define the instrument parameters
    # pathlength = 6766.56 # cm
    pathlength = distance
    # laser current in miliamps
    # max and min laser power in sweep
    I_start, I_end = 33.9, 43.1
    # reference current and wavelength from lab measurements
    I1, λ1 = 69.11, 1653.678
    I2, λ2 = 96.61, 1653.707
    #165.72*.5411
    # I1, λ1 = 73.60, 1653.678
    # I2, λ2 = 92.09, 1653.707
    t0 = 0
    t1 = 0.05
    line_center_loc = length(measurement.intensity) - argmin(measurement.intensity)
    t_center = (line_center_loc/length(measurement.intensity))*0.05
    λ_center = 1653.73
    # extrapolate the wavenumber range of the instrument from lab reference-points
    λ_start, λ_end= calc_wavelength_range(t0, t1, λ_center, t_center)
    @show λ_start, λ_end
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
    βloss = 0.99

    ## define laser modulation parameters 
    # convert amplitude of fast-modulation from nanometers to 1/cm
    # experimentally derived parameter for fast sweep 
    mod_amplitude = 0.0235/2 # nm
    mod_amplitude = convert_amplitude(min_wavenumber, max_wavenumber, mod_amplitude) # cm⁻¹
    @show mod_amplitude
    mod_freq = 9.0e3 # fast sweep in hz
    num_samples = (2*length(measurement.intensity))
    slow_sweep_period = 0.1 #sec per measurement sweep 
    sampling_rate = num_samples / slow_sweep_period # samples / sec
    # sampling_rate = 100.0e3 # samples / sec

    # length of the vector for one full period
    tuning_rate = (max_wavenumber - min_wavenumber) / slow_sweep_period
    # the time vector for one full sweep period
    # t = range(0, stop=slow_sweep_period, length=num_samples)

    resistor_cell = 50
    responsivity_cell = 0.9
    gain_room_pd = 4.75e3
    responsivity_room = 0.95
    cell_scale = measurement.cell/(resistor_cell*responsivity_cell)

    # parameters for modulation of laser light power
    I0 = mean(measurement.room) # avg laser intensity
    @show I0
    @show sampling_rate

    ## calculate the amplitude` of the fast-sweep modulation with the data
    # Take a 9-k modulation frequency snippet
    fast_sweep_period = 1.0 / mod_freq
    # take the first 9-k modulation frequency snippet
    fast_sweep = measurement.room[1:round(Int, 1.5*fast_sweep_period * sampling_rate)]


    min_room = minimum(measurement.room)
    # calculate the amplitude of the fast-sweep modulation
    i0 = ((maximum(fast_sweep) - minimum(fast_sweep)) / 2)
    I0_triangle_amp = (((maximum(measurement.room)-i0)-(minimum(measurement.room)+i0)))
    i2 = 0.0 # ignore the nonlinearities for the moment
    psi1 = 0.0 # ignore phase shift rn
    psi2 = 0.0 # ignore phase shift rn

    sensitivity = 0.4
    # gain = (10/sensitivity)*(pi/(2*sqrt(2)))
    lockin_gain = 40    #dB
    gain = (10^(lockin_gain/20))

    # build the laser power struct
    laser_power = LaserPower(I0=I0_triangle_amp, i0=i0, i2=i2, psi1=psi1, psi2=psi2, min_room=min_room)

    # create the instrument object
    instrument = Instrument(wavenumber_range=wavenumber_range, avg_laser_intensity=I0,
    slow_sweep_period=slow_sweep_period, tuning_rate=tuning_rate,
    mod_amplitude=mod_amplitude, mod_freq=mod_freq, num_samples=num_samples, pathlength=pathlength, 
    sampling_rate=sampling_rate, laser_power=laser_power, gain = gain
    )
    return instrument, laser_power

end