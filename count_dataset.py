import os

# Check both dataset locations
datasets = [
    ("New_Model Dataset", r"d:\SEMESTER 8 COURSES\Thesis\New_Model\Dataset"),
    ("Thesis Project Dataset", r"d:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset"),
]

for name, base in datasets:
    print(f"\n{'='*50}")
    print(f"  {name}: {base}")
    print(f"{'='*50}")
    
    if not os.path.exists(base):
        print("  NOT FOUND")
        continue
    
    # Direct class folders
    for cls in ['Normal', 'Pneumonia', 'Tuberculosis']:
        path = os.path.join(base, cls)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"  {cls}: {count}")
    
    # Split folders
    for split in ['train', 'validation', 'test', 'final', 'processed', 'raw']:
        split_path = os.path.join(base, split)
        if os.path.exists(split_path):
            print(f"\n  --- {split} ---")
            for cls in ['Normal', 'Pneumonia', 'Tuberculosis']:
                path = os.path.join(split_path, cls)
                if os.path.exists(path):
                    count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                    print(f"    {cls}: {count}")
                else:
                    print(f"    {cls}: (folder not found)")
