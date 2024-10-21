import csv
import matplotlib.pyplot as plt
import datetime

with open('loc_coords.txt', 'r') as in_file:
    stripped = (line.strip()[:-1] for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    lines = [(line[0], line[1], line[2], line[3]) for line in lines]
    timestamps = [datetime.datetime.strptime(line[0].split(".")[0], "%Y-%m-%d %H:%M:%S") for line in lines]
    locs_yaw = [float(line[1]) for line in lines]
    locs_pitch = [float(line[2]) for line in lines]
    intensities = [float(line[3]) for line in lines]
print(locs_yaw)
f, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(timestamps, locs_yaw)
ax2.plot(timestamps, locs_pitch)
ax3.plot(timestamps, intensities)

f.savefig("test.png")