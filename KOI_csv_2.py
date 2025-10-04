import os
import pandas as pd
import lightkurve as lk
import time

#config
koi_list_file = "lista.txt"
mission = "Kepler"
output_dir = "KOI_lightcurves_fin"

os.makedirs(output_dir, exist_ok=True)

koi_table = pd.read_csv(koi_list_file, sep='\t', header=None, names=['kepid', 'disposition'])
koi_table['kepid'] = koi_table['kepid'].astype(str).str.strip()
koi_table['disposition'] = koi_table['disposition'].astype(str).str.strip().str.upper()

koi_table = koi_table.drop_duplicates(subset=['kepid', 'disposition'])

print(f"Loaded {len(koi_table)} unique KOIs from input file")

for idx, row in koi_table.iterrows():
    kepid = row['kepid']
    disposition = row['disposition']

    try:
        if disposition not in ["CONFIRMED", "FALSE POSITIVE"]:
            print(f"\nSkipping {kepid} (disposition = {disposition})")
            continue

        csv_path = os.path.join(output_dir, f"{kepid}.csv")
        if os.path.exists(csv_path):
            print(f"\nSkipping {kepid} (CSV already exists)")
            continue

        print(f"\nProcessing {kepid} (disposition = {disposition})...")

        # --- Stopwatch: search ---
        t0 = time.time()
        sr = lk.search_lightcurve(kepid, mission=mission)
        t1 = time.time()
        print(f"  Search done in {t1 - t0:.2f} s")

        if len(sr) == 0:
            print(f"  No lightcurves for {kepid}, SKIP!")
            continue
        print(f"  Found {len(sr)} lightcurves for {kepid}")

        # --- Stopwatch: download ---
        t2 = time.time()
        lc = sr[0].download(download_dir="cached_data", cutout_size=None, cache=True)
        t3 = time.time()
        if lc is None:
            print(f"  Failed to download {kepid}, SKIP!")
            continue
        print(f"  Downloaded {kepid} in {t3 - t2:.2f} s")

        # --- Stopwatch: normalize ---
        t4 = time.time()
        lc = lc.normalize()
        t5 = time.time()
        print(f"  Normalized {kepid} in {t5 - t4:.2f} s")

        # --- Stopwatch: save CSV ---
        t6 = time.time()
        df = pd.DataFrame({
            "time": lc.time.value,
            "flux": lc.flux.value,
            "flux_err": lc.flux_err.value,
            "kepid": kepid,
            "disposition": disposition
        })
        df.to_csv(csv_path, index=False)
        t7 = time.time()
        print(f"  Saved CSV in {t7 - t6:.2f} s")

        print(f"  Total time for {kepid}: {t7 - t0:.2f} s")

    except Exception as e:
        print(f"  Error processing {kepid}: {e}")
