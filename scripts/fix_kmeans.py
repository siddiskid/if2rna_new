import h5py
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ref_file = "data/metadata/reference.csv"
feature_dir = Path("data/processed/features")
n_clusters = 100

df = pd.read_csv(ref_file)

for _, row in tqdm(df.iterrows(), total=len(df)):
    wsi = row['wsi_file_name']
    project = row['tcga_project']
    
    h5_path = feature_dir / project / wsi / f"{wsi}.h5"
    
    if not h5_path.exists():
        print(f"Missing: {h5_path}")
        continue
    
    with h5py.File(h5_path, 'r+') as f:
        if 'cluster_features' in f:
            print(f"Skipping {wsi} (already has cluster_features)")
            continue
            
        if 'uni_features' not in f:
            print(f"Missing uni_features in {wsi}")
            continue
        
        # Get uni features
        features = f['uni_features'][:]
        
        # Run k-means
        kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=99)
        kmeans.fit(features)
        
        # Save cluster centers
        f.create_dataset('cluster_features', data=kmeans.cluster_centers_)
        print(f"✓ {wsi}: {kmeans.cluster_centers_.shape}")

print("Done!")
