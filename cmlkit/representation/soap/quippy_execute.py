# the script that actually computes SOAP. Called by `quippy_interface.py`.
import os

print(
    "This is the output of the run_quippy.py script. Printing the environment variables for easier debugging."
)
print(os.environ)

import quippy
import ase
import numpy as np


with open("quippy_config.txt", "r") as f:
    config = f.read()

all_ase = ase.io.iread("data.traj")

descs = []
cutoffs = []
for at_ase in all_ase:
    at = quippy.Atoms(at_ase)
    desc = quippy.descriptors.Descriptor(config)
    at.set_cutoff(desc.cutoff())
    at.calc_connect()

    desc_out = desc.calc(at)
    descs.append(np.round(desc_out["descriptor"], 20))
    cutoffs.append(desc_out["cutoff"])

np.save("out", descs)  # automatically saves as dtype=object
