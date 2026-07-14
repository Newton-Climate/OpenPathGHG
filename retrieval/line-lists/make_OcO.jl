using vSmartMOM.Absorption, JLD2, HDF5, Interpolations, StatsBase 



"""
Build an interpolation model from the JPL OCO cross-section grid.

The output is saved as a vSmartMOM `InterpolationModel` for downstream
retrieval and simulation workflows.
"""
filename = "OcO_spectra.hdf"
        
    fid = h5open(filename, "r")
    p = read(fid, "Pressure") * 0.01 # convert Pa to mbar
    T = read(fid, "Temperature")[:,1];
    ν = read(fid, "Wavenumber")[19002:end]
    σ = read(fid, "Gas_02_Absorption")[19002:end, 3,:,:]; # cross-sections
    broadener = read(fid, "Broadener_01_VMR")
    close(fid)
    
 σ = permutedims(σ, [1, 3, 2]) 
    ### Build regularized coordinate axes and interpolation object.
    p_min, p_max = p[1], p[end];
    δp = mean(diff(p));
    T_min, T_max = T[1], T[end];
    δT= mean(diff(T));
    ν_min, ν_max = ν[1], ν[end];
    δν = mean(diff(ν))
    broadener_min, broadener_max = broadener[1], broadener[end];
    δ_broadener = mean(diff(broadener));
    
    # Create axis ranges and interpolation grid metadata.
    p = p_min:δp:p_max;
    T = T_min:δT:T_max;
    ν = ν_min:δν:ν_max;
    broadener = broadener_min:δ_broadener:broadener_max;
    grid = (ν, broadener, T, p);
    # Alternative gridded interpolation retained for reference:
    # sitp = interpolate(grid, σ, Gridded(Linear()))
    itp = interpolate(σ, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, ν, p, T);
itp_model = InterpolationModel(itp=sitp, mol=2, iso=1, ν_grid=ν, p_grid=p, t_grid=T)
@save "OCO_12CO2_weakBand_006VMR.jld2" itp_model
