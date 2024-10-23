Software installs:
https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10285 Observe and control motors by themselves
https://digilent.com/shop/software/digilent-waveforms/ Observe scope - spectrum analyzer, select centered at 9 kHz, sometimes set scale to linear instead of dB
https://github.com/Newton-Climate/OpenPathGHG/tree/main/instrument_control The software, use requirements.txt to set up all pip dependencies, py folder contains other imported dependencies
Useful: https://pylablib.readthedocs.io/en/latest/.apidoc/pylablib.devices.Thorlabs.html#pylablib.devices.Thorlabs.kinesis.KinesisMotor.get_scale information on Thorlabs device api


Physical Setup:
Attach scope 1+ and 1- to detector poles, then potentially a preamp in between. Digilent USB plugs into computer, along with thorlabs kcube mount.

Running code (and rough consensus):
Run app.py to bring up a UI with 3 buttons and 2 graphs.
Keep Beam Aligned continually runs AdjustBeams until you tell it to stop. Note that if the button is clicked 2-3 times in quick succession this would cause a thread error, so I have locked out the button until the alignment routine has safely ended.
Map Beam’s Physical Field runs scans over a rectangular area based on how quickly the signal falls off in yaw and pitch in both directions, and maps it to a file which it spits out.
Calibrate Device calibrates the sample size of each checkVal call, and then how big a step the motors should take in both the yaw and pitch directions. There’s probably room to improve or speed up this routine, but it works fine for now, or you could do without it.
Graph 1 displays the last spectrum observed. Since we only observe spectrums when the program calls for it, the graph is frozen ex. during motor movements.
Graph 2 displays each movement the motors have made. Note that this is not the same as the movements spit out to loc_coords, which are the final movements the motor makes after finishing a single alignment step.
Calling interpretcoords.py in the data folder converts loc_coords into a graph showing yaw, pitch, and amplitude over time, and is useful for debugging long runs.
