import os
import pandas as pd
import lightkurve as lk

# config
koi_list_file = "Archive/koi_list.txt"
mission = "Kepler"
output_dir = "Archive/KOI_lightcurves"

os.makedirs(output_dir, exist_ok=True)

with open(koi_list_file, "r") as f:
    kois = [line.strip() for line in f if line.strip()]

print(f"Found {len(kois)} KOIs")

for koi in kois:
    try:
        csv_path = os.path.join(output_dir, f"{koi}.csv")
        if os.path.exists(csv_path):
            print(f"\nSkipping {koi} (CSV already exists)")
            continue

        print(f"\nProcessing {koi}...")
        
        sr = lk.search_lightcurve(koi, mission=mission)
        if len(sr) == 0:
            print(f"  No lightcurves of {koi}, SKIP!")
            continue
        else:
            print(f"  Found {len(sr)} lightcurves for {koi}")
            continue
        
        # Efficient download without extra pixel data
        lc = sr[0].download(download_dir="cached_data", cutout_size=None, cache=True)
        print(f"  Downloaded {koi} lightcurve")
        if lc is None:
            print(f"  Failed @ {koi}, SKIP!")
            continue

        lc = lc.normalize()
        print(f"  Normalized {koi} lightcurve")

        df = pd.DataFrame({
            "time": lc.time.value,  
            "flux": lc.flux.value,
            "flux_err": lc.flux_err.value
        })
        print(f"  Prepared DataFrame for {koi}")

        df.to_csv(csv_path, index=False)
        print(f"  Saved {koi} lightcurve @ {csv_path}")

    except Exception as e:
        print(f"  Error processing {koi}: {e}")
