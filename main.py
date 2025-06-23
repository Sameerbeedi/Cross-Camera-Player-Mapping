import os

print("Step 1: Running YOLO detection...")
os.system("python detectplayers.py")

print("Step 2: Extracting visual features...")
os.system("python extract_features.py")

print("Step 3: Matching players across videos...")
os.system("python matching.py")

print("Step 4: Visualizing annotated videos...")
os.system("python visualize.py")

print("\n✅ Pipeline completed successfully!")