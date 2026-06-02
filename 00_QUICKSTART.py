"""
QUICK START GUIDE
Run this first before anything else
"""

import subprocess
import sys
from pathlib import Path

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║            RADON PREDICTION SYSTEM - IRBID CITY, JORDAN                      ║
║                         Quick Start Guide                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("\n📋 STEP 1: Checking Python version...")
version = sys.version_info
if version.major >= 3 and version.minor >= 7:
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
else:
    print(f"✗ Python 3.7+ required (you have {version.major}.{version.minor})")
    sys.exit(1)

print("\n📦 STEP 2: Installing dependencies...")
requirements_file = Path('requirements.txt')
if requirements_file.exists():
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt',
                       '--quiet'], check=True)
        print("✓ All dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Some dependencies may not have installed correctly")
        print(f"  Error: {e}")
else:
    print("✗ requirements.txt not found")
    print("  Installing essential packages manually...")
    packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'scipy', 'folium']
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + packages + ['--quiet'], 
                      check=True)
        print("✓ Essential packages installed")
    except Exception as e:
        print(f"⚠ Error installing packages: {e}")

print("\n🗂️  STEP 3: Checking project structure...")
required_files = [
    '1_data_cleaning_and_preparation.py',
    '2_ml_model_development.py',
    '3_interactive_maps.py',
    '4_geological_analysis_and_reporting.py',
    '5_master_execution.py',
]

missing = []
for file in required_files:
    if Path(file).exists():
        print(f"✓ {file}")
    else:
        print(f"✗ {file} (MISSING)")
        missing.append(file)

if missing:
    print(f"\n⚠ Warning: {len(missing)} files are missing")

print("\n" + "="*80)
print("READY TO START ANALYSIS!")
print("="*80)

print("""
Choose one of the following options:

Option A - RUN EVERYTHING (Recommended for first time)
───────────────────────────────────────────────────────
  python 5_master_execution.py

  This will automatically:
  1. Clean and prepare data
  2. Train machine learning models
  3. Generate interactive maps
  4. Perform geological analysis
  5. Create all visualizations

  ⏱️  Estimated time: 5-10 minutes


Option B - RUN INDIVIDUAL STEPS
────────────────────────────────
  Step 1 - Data Cleaning:
    python 1_data_cleaning_and_preparation.py

  Step 2 - Model Development:
    python 2_ml_model_development.py

  Step 3 - Interactive Maps:
    python 3_interactive_maps.py

  Step 4 - Geological Analysis:
    python 4_geological_analysis_and_reporting.py


Option C - QUICK TEST
──────────────────────
  Just run Step 1 to see if everything works:
    python 1_data_cleaning_and_preparation.py


📊 OUTPUT INFORMATION
─────────────────────

After running the analysis, you'll have:

✓ DATA FILES
  - radon_data_cleaned.csv (cleaned dataset)

✓ VISUALIZATIONS (PNG images)
  - feature_importance.png
  - model_comparison.png
  - geological_analysis.png
  - correlation_analysis.png
  - regional_analysis.png
  - And more...

✓ INTERACTIVE MAPS (HTML files - open in browser)
  - heatmap_radon_measurements.html
  - prediction_map_irbid.html
  - region_comparison_map.html
  - geological_analysis_map.html

✓ ANALYSIS RESULTS
  - model_comparison_results.csv
  - Console output with statistics


🚀 GET STARTED NOW
──────────────────

Run the complete analysis:

    python 5_master_execution.py


❓ NEED HELP?
─────────────
1. Check README.md for detailed information
2. Review individual script comments
3. Check console output for error messages
4. Verify all dependencies are installed

""")

response = input("\n▶ Start analysis now? (yes/no): ").strip().lower()

if response in ['yes', 'y', '1']:
    print("\n🚀 Starting analysis...")
    print("="*80 + "\n")
    try:
        subprocess.run([sys.executable, '5_master_execution.py'])
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
else:
    print("\n✓ Setup complete! Run '5_master_execution.py' when ready.")
    print("  Happy analyzing! 🎉")
