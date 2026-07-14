using vSmartMOM.Absorption, JLD2

### Build and save interpolation lookup tables for selected species.
#
# Most blocks are intentionally left commented to enable ad-hoc table
# generation for specific molecules and databases.

# define the grids 
temperatures = 180.0:10.0:310.0
pressures = 180.0:10.0:1100.0
ν_grid = 6076.0:0.001:6078.0


#hit16_co2 = read_hitran("hit16_12CO2.par", mol=2, iso=1, ν_min=6000, ν_max=6400)
#itp_model = make_interpolation_model(hit16_co2, Voigt(), ν_grid, pressures, temperatures, wing_cutoff = 40, CEF=HumlicekWeidemann32SDErrorFunction())
#@save "/net/fluo/data1/data/NIST/hit16_12CO2.jld2" itp_model


#hit16_ch4 = read_hitran("hit16_12CH4.par", mol=6, iso=1, ν_min=6000, ν_max=6400)
#itp_model = make_interpolation_model(hit16_ch4, Voigt(), ν_grid, pressures, temperatures, wing_cutoff = 40, CEF=HumlicekWeidemann32SDErrorFunction())
#@save "/net/fluo/data1/data/NIST/hit16_12CH4.jld2" itp_model


#hit16_h2o = read_hitran("hit16_H2O.par", mol=1, iso=1, ν_min=6000, ν_max=6400)
#itp_model = make_interpolation_model(hit16_h2o, Voigt(), ν_grid, pressures, temperatures, wing_cutoff = 40, CEF=HumlicekWeidemann32SDErrorFunction())
#@save "/net/fluo/data1/data/NIST/hit16_H2O.jld2" itp_model

#hit16_hdo = read_hitran("hit16_HDO.par", mol=1, iso=4, ν_min=6000, ν_max=6400)
#itp_model = make_interpolation_model(hit16_hdo, Voigt(), ν_grid, pressures, temperatures, wing_cutoff = 40, CEF=HumlicekWeidemann32SDErrorFunction())
#@save "/net/fluo/data1/data/NIST/hit16_HDO.jld2" itp_model


#hit08_ch4 = read_hitran("hit08_12CH4.par", mol=6, iso=1, ν_min=6000, ν_max=6400)
#itp_model = make_interpolation_model(hit08_ch4, Voigt(), ν_grid, pressures, temperatures, wing_cutoff = 40, CEF=HumlicekWeidemann32SDErrorFunction())
#@save "/net/fluo/data1/data/NIST/hit08_12CH4.jld2" itp_model


### TCCON-derived tables
#ν_grid = 6200.0:0.01:6400.0
tccon_co2 = read_hitran("tccon_2020.par", mol=2, iso=1, ν_min=6000, ν_max=6400)
itp_model = make_interpolation_model(tccon_co2, Voigt(), ν_grid, pressures, temperatures, wing_cutoff = 40, CEF=HumlicekWeidemann32SDErrorFunction())
@save "/net/fluo/data1/data/NIST/tccon_12CO2.jld2" itp_model


tccon_ch4 = read_hitran("tccon_2020.par", mol=6, iso=1, ν_min=6000, ν_max=6400)
itp_model = make_interpolation_model(tccon_ch4, Voigt(), ν_grid, pressures, temperatures, wing_cutoff = 40, CEF=HumlicekWeidemann32SDErrorFunction())
@save "/net/fluo/data1/data/NIST/tccon_12CH4.jld2" itp_model


tccon_h2o = read_hitran("tccon_2020.par", mol=1, iso=1, ν_min=6000, ν_max=6400)
itp_model = make_interpolation_model(tccon_h2o, Voigt(), ν_grid, pressures, temperatures, wing_cutoff = 40, CEF=HumlicekWeidemann32SDErrorFunction())
@save "/net/fluo/data1/data/NIST/tccon_H2O.jld2" itp_model

tccon_hdo = read_hitran("tccon_2020.par", mol=49, iso=1, ν_min=6000, ν_max=6400)
itp_model = make_interpolation_model(tccon_hdo, Voigt(), ν_grid, pressures, temperatures, wing_cutoff = 40, CEF=HumlicekWeidemann32SDErrorFunction())
@save "/net/fluo/data1/data/NIST/tccon_HDO.jld2" itp_model
