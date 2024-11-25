# functions for reading data files

function find_peaks(df, i)
    cell_num = 4*i - 3
    room_num = 4*i - 2
    lock_num = 4*i - 1
    
    cell = df[Symbol("Column$cell_num")]
    lockin = df[Symbol("Column$lock_num")]
    room = df[Symbol("Column$room_num")]
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


function find_peaks(df, i)
    # cell = df.cell
    # lockin = df.lockin
    # room = df.room
    cell_num = 4*i - 3
    room_num = 4*i - 2
    lock_num = 4*i - 1

    cell = df[!,Symbol("Column$cell_num")]
    lockin = df[!,Symbol("Column$lock_num")]
    room = df[!,Symbol("Column$room_num")]

    L = length(cell)
    half_point = div(L, 2)
    quarter_length = div(L, 4)
    delta = 50
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
        cell_rise = cell[min_pt1:max_pt]
        falling = lockin[max_pt:min_pt2]
        room_fall = room[max_pt:min_pt2]
        cell_fall = cell[max_pt:min_pt2]
    else
        # Assume two maxima
        min_pt = max_pt1 + argmin(cell[max_pt1:max_pt2])
        rising = lockin[min_pt:max_pt2]
        room_rise = room[min_pt:max_pt2]
        cell_rise = cell[min_pt:max_pt2]
        falling = lockin[max_pt1:min_pt]
        room_fall = room[max_pt1:min_pt]
        cell_fall = cell[max_pt1:min_pt]
    end
    t_data = range(0, 0.1, length=length(cell))

    return rising, room_rise, cell_rise, t_data
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

function average_measurements(file_path, col_names, num_iter_meas, wavenumber_range=wavenumber_range, p=p, T=T)
    iteration = true

    for k in num_iter_meas
        global measurement = read_measurement(file_path, col_names, k, wavenumber_range=wavenumber_range, p=p, T=T)
        
        if iteration == true
            global average_intensity = measurement.intensity
            global average_room = measurement.room
            global average_cell = measurement.cell

            iteration = false
        else
           
            L1 = length(measurement.intensity)
            L2 = length(average_intensity)
            
            ΔL = abs(L1-L2)

            if mod(ΔL,2) == 0
                #even
                n_start = n_end = Int(ΔL/2)
            end

            if mod(ΔL,2) == 1
                #odd
                n_start = Int((ΔL/2)-0.5)
                n_end = Int((ΔL/2)+0.5)
            end

            if L1 > L2 
                #trimming new measurement        
                measurement.intensity = measurement.intensity[1+n_start:end-n_end]
                measurement.room = measurement.room[1+n_start:end-n_end]
                measurement.cell = measurement.cell[1+n_start:end-n_end]
            end

            if L1 < L2
                #trimming average measurement
                average_intensity = average_intensity[1+n_start:end-n_end]
                average_room = average_room[1+n_start:end-n_end]
                average_cell = average_cell[1+n_start:end-n_end]

            end
            

            global average_intensity = (average_intensity .+ measurement.intensity) / 2
            global average_room = (average_room .+ measurement.room) / 2
            global average_cell = (average_cell .+ measurement.cell) / 2


        end
        
    end
    print("beep0")
    measurement.intensity = average_intensity
    measurement.room = average_room
    measurement.cell = average_cell

    print("beep")
    return measurement
end

function average_measurements(file_path, col_names, num_iter_meas, wavenumber_range=wavenumber_range, p=p, T=T)
    iteration = true

    for k in num_iter_meas
        global measurement = read_measurement(file_path, col_names, k, wavenumber_range=wavenumber_range, p=p, T=T)
        
        if iteration == true
            global average_intensity = measurement.intensity
            global average_room = measurement.room
            global average_cell = measurement.cell

            iteration = false
        else
           
            L1 = length(measurement.intensity)
            L2 = length(average_intensity)
            
            ΔL = abs(L1-L2)

            if mod(ΔL,2) == 0
                #even
                n_start = n_end = Int(ΔL/2)
            end

            if mod(ΔL,2) == 1
                #odd
                n_start = Int((ΔL/2)-0.5)
                n_end = Int((ΔL/2)+0.5)
            end

            if L1 > L2 
                #trimming new measurement        
                measurement.intensity = measurement.intensity[1+n_start:end-n_end]
                measurement.room = measurement.room[1+n_start:end-n_end]
                measurement.cell = measurement.cell[1+n_start:end-n_end]
            end

            if L1 < L2
                #trimming average measurement
                average_intensity = average_intensity[1+n_start:end-n_end]
                average_room = average_room[1+n_start:end-n_end]
                average_cell = average_cell[1+n_start:end-n_end]

            end
            

            global average_intensity = (average_intensity .+ measurement.intensity) / 2
            global average_room = (average_room .+ measurement.room) / 2
            global average_cell = (average_cell .+ measurement.cell) / 2


        end
        
    end
    print("beep0")
    measurement.intensity = average_intensity
    measurement.room = average_room
    measurement.cell = average_cell

    print("beep")
    return measurement
end

function read_measurement(df::DataFrame,
    col_names::Vector{String},
    i::Integer;
    random_noise_sd=nothing,
    wavenumber_range::Tuple{FT, FT}=(6047.5, 6047.75),
    p::FT=1000.0,
    T::FT=290.0) where FT <: AbstractFloat

    # read the CSV file as a dataframe
    # df = CSV.read(file_path, DataFrame, delim=',', header=["cell", "room", "lockin"])
    # df = CSV.read(file_path, DataFrame, delim=',', header=col_names)
    

    # filter out the rising parts of the wavelength sweep
    (lockin_rise, environmental_rise, cell_rise, t_data) = find_peaks(df, i)
    half_period::Int = length(lockin_rise) 
    # ind = collect(1:half_period)
    # the spectral grid of the instrument
    # NN: note that this may have to be switched out
     # with something more dynamic and accounts for fast-sweep
    instrument_grid = collect(range(wavenumber_range[1], wavenumber_range[2], length=half_period))
    
    
    # p_cell = plot(t_data, df.Column1, label="cell")
    # p_room = plot(t_data, df.Column2, label="room")
    # p_lock= plot(t_data, df.Column3, label="lock")
    # xlabel!("wave-number")
    # ylabel!("intensity")
    # plot(p_cell, p_room, p_lock, layout=(3,1), legend=:bottomright)
    # savefig("raw_data.png")

    # read the timestamp from the file
    timestamp, machine_time = _get_timestamp(file_path)

        # create the signal vector
        shift = 0.0
        signal = lockin_rise

        # calculate the 1-σ instrument noise
        if random_noise_sd == nothing
            random_noise_sd = calc_noise(signal, instrument_grid, 6047.5, 6047.75, poly_degree=2)
        end
        
    # create the measurement struct
    return FreqModulationMeasurement(
        intensity=reverse(signal),
        grid=instrument_grid,
        cell=cell_rise,
        room=environmental_rise,
        temperature=T,
        pressure=p,
        time=timestamp,
        machine_time=machine_time,
        num_averaged_measurements=1,
        σ²=random_noise_sd^2
    )
end
