import os
import lightkurve as lk
import matplotlib.pyplot as plt

#config
target_name = "TOI-181"
mission = "TESS"
output_dir = "figures"
output_file = "toi181_lightcurve.png"

os.makedirs(output_dir, exist_ok=True)

print(f"Searching for {target_name} from {mission}...")
sr = lk.search_lightcurve(target_name, mission=mission)
# print(f"Found {len(sr)} lightcurves.")
print(sr[0].ra)

print("Downloading...")
# lc = sr[0].download()
# lc = lc.normalize()
# print("Done & normalized")
#
# plt.figure(figsize=(10,5))
# lc.plot()
# plt.title(f"{target_name} TESS Lightcurve")
# plt.xlabel("[BTJD]")
# plt.ylabel("Normalized Flux")
# plt.grid(True)
#
# save_path = os.path.join(output_dir, output_file)
# plt.savefig(save_path)
# plt.close()
# print(f"Lightcurve plot @ {save_path}")
