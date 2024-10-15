# functions for reading data files

function find_peaks(df)
    cell = df.cell
    lockin = df.lockin
    room = df.room
    L = length(cell)
    half_point = div(L, 2)
    quarter_length = div(L, 4)

    # Compute minima and maxima and their indices for each half
    min1, max1 = minimum(cell[1:half_point]), maximum(cell[1:half_point])
    min_pt1, max_pt1 = argmin(cell[1:half_point]), argmax(cell[1:half_point])

    min2, max2 = minimum(cell[half_point+1:end]), maximum(cell[half_point+1:end])
    min_pt2, max_pt2 = argmin(cell[half_point+1:end]) + half_point, argmax(cell[half_point+1:end]) + half_point

    # Determine if there is a peak in the center by comparing the positions
    delta_max_pt, delta_min_pt = abs(max_pt2 - max_pt1), abs(min_pt2 - min_pt1)

    if delta_max_pt <= quarter_length || delta_min_pt <= quarter_length
        max_center, min_center = abs(half_point - max_pt1), abs(half_point - min_pt1)
        truth_state = max_center < min_center
    else
        truth_state = abs(max2 - max1) > abs(min2 - min1)
    end

    if truth_state
        # Assume two minima
        max_pt = min_pt1 + argmax(cell[min_pt1:min_pt2])
        rising = lockin[min_pt1:max_pt]
        room_rise = room[min_pt1:max_pt]
        falling = lockin[max_pt:min_pt2]
        room_fall = room[max_pt:min_pt2]
    else
        # Assume two maxima
        min_pt = max_pt1 + argmin(cell[max_pt1:max_pt2])
        rising = lockin[min_pt:max_pt2]
        room_rise = room[min_pt:max_pt2]
        falling = lockin[max_pt1:min_pt]
        room_fall = room[max_pt1:min_pt]
    end

    return rising, room_rise
end



"""
    struct FreqModulationMeasurement{FT}
"""
    
Base.@kwdef mutable struct FreqModulationMeasurement{FT} <: SpectralFits.AbstractMeasurement
    intensity::Vector{FT}
    grid::Vector{FT}
    cell::Vector{FT}
    room::Vector{FT}
    temperature::Union{FT, Vector{FT}}
    pressure::Union{FT, Vector{FT}}
    time::Dates.DateTime
    machine_time::FT
    num_averaged_measurements::Int
    σ²::FT
end


function find_peaks(df)
    cell = df.cell
    lockin = df.lockin
    room = df.room
    L = length(cell)
    half_point = div(L, 2)
    quarter_length = div(L, 4)

    # Compute minima and maxima and their indices for each half
    min1, max1 = minimum(cell[1:half_point]), maximum(cell[1:half_point])
    min_pt1, max_pt1 = argmin(cell[1:half_point]), argmax(cell[1:half_point])

    min2, max2 = minimum(cell[half_point+1:end]), maximum(cell[half_point+1:end])
    min_pt2, max_pt2 = argmin(cell[half_point+1:end]) + half_point, argmax(cell[half_point+1:end]) + half_point

    # Determine if there is a peak in the center by comparing the positions
    delta_max_pt, delta_min_pt = abs(max_pt2 - max_pt1), abs(min_pt2 - min_pt1)

    if delta_max_pt <= quarter_length || delta_min_pt <= quarter_length
        max_center, min_center = abs(half_point - max_pt1), abs(half_point - min_pt1)
        truth_state = max_center < min_center
    else
        truth_state = abs(max2 - max1) > abs(min2 - min1)
    end

    if truth_state
        # Assume two minima
        max_pt = min_pt1 + argmax(cell[min_pt1:min_pt2])
        rising = lockin[min_pt1:max_pt]
        room_rise = room[min_pt1:max_pt]
        falling = lockin[max_pt:min_pt2]
        room_fall = room[max_pt:min_pt2]
    else
        # Assume two maxima
        min_pt = max_pt1 + argmin(cell[max_pt1:max_pt2])
        rising = lockin[min_pt:max_pt2]
        room_rise = room[min_pt:max_pt2]
        falling = lockin[max_pt1:min_pt]
        room_fall = room[max_pt1:min_pt]
    end

    return rising, room_rise
end


"""function to normalize lockin signal with environmental signal"""
function normalize_lockin_signal(snippetToFit, peakToCompare)
    # Generate x values corresponding to the indices of the snippet
    x = collect(1:length(snippetToFit))
    
    # Fit a linear model to the snippet
    A = hcat(x, ones(length(x)))  # Design matrix with intercept
    coeff = A \ snippetToFit      # Perform linear regression via least squares
    
    slope, intercept = coeff[1], coeff[2]
    
    # Calculate the line of best fit
    lineOfBestFit = slope .* x .+ intercept
    
    # Find the voltage value at the minimum of peakToCompare
    minIndex = argmin(peakToCompare)
    voltVal = lineOfBestFit[minIndex]

    # Optional plotting (uncomment if you want to visualize the fit and data)
    # using Plots
    # plot(x, snippetToFit, label="Original Data")
    # plot!(x, lineOfBestFit, label="Line of Best Fit", linestyle=:dash)

    return voltVal
end


# notes 

# read in measurements from file and create the measurment struct
function _get_timestamp(file_path::String)
    # read the timestamp from the file time of creation
    timestamp = stat(file_path).ctime
    # convert to DateTime
    return Dates.unix2datetime(timestamp), timestamp
end



function read_measurement(filename::String;
    random_noise_sd=nothing,
    wavenumber_range::Tuple{FT, FT}=(6047.5, 6047.75),
    p::FT=1000.0,
    T::FT=290.0) where FT <: AbstractFloat

    # read the CSV file as a dataframe
    df = CSV.read(file_path, DataFrame, delim=',', header=["cell", "room", "lockin"])
    # filter out the rising parts of the wavelength sweep
    (lockin_rise, environmental_rise) = find_peaks(df)
    half_period::Int = length(lockin_rise) 
    ind = collect(1:half_period)

    # the spectral grid of the instrument
    # NN: note that this may have to be switched out
     # with something more dynamic and accounts for fast-sweep
    instrument_grid = collect(range(wavenumber_range[1], wavenumber_range[2], length=half_period))

    # read the timestamp from the file
    timestamp, machine_time = _get_timestamp(file_path)

        # create the signal vector
        shift = 0.0
        signal = lockin_rise[ind]

        # calculate the 1-σ instrument noise
        if random_noise_sd == nothing
            random_noise_sd = calc_noise(signal, instrument_grid, 6047.5, 6047.75, poly_degree=2)
        end
        
    # create the measurement struct
    return FreqModulationMeasurement(
        intensity=signal,
        grid=instrument_grid[ind],
        cell=df.cell[ind],
        room=df.room[ind],
        temperature=T,
        pressure=p,
        time=timestamp,
        machine_time=machine_time,
        num_averaged_measurements=1,
        σ²=random_noise_sd^2
    )
end
