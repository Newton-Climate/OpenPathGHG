using vSmartMOM.Absorption, JLD2, Interpolations, NPZ, StatsBase

"""
    arr2range(arr)

Convert an approximately uniform numeric vector into a Julia range using
the first value, mean step, and last value.
"""
function arr2range(arr)
    return arr[1]:mean(diff(arr)):arr[end]
    
end
    
ds = npzread("HT_12CH4.npz")
# Reconstruct axes from stored arrays.
p = arr2range(ds["p"])
T = arr2range(ds["T"])
nu = arr2range(npzread("grid.npy"))
#nu = arr2range(ds["nu"])
sigma = ds["sigma"]

# Build interpolation object and package it into an InterpolationModel.
itp = interpolate(sigma, (BSpline(Linear()),BSpline(Cubic(Line(OnGrid()))),BSpline(Cubic(Line(OnGrid())))))
itp = interpolate(sigma, BSpline(Cubic(Line(OnGrid()))))

itp_model = InterpolationModel(itp, 6, 1,  nu, p, T)
@save "HT_12CH4.jld2" itp_model
