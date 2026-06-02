"""
STEP 1: DATA CLEANING AND PREPARATION
This script cleans the raw radon data and exports it as a clean CSV
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def clean_elevation(elev_str):
    """Clean elevation strings (handle ranges like '800-900')"""
    if pd.isna(elev_str):
        return np.nan
    
    elev_str = str(elev_str).strip()
    
    # Handle ranges (e.g., "800-900")
    if '-' in elev_str and 'i' not in elev_str:
        try:
            parts = elev_str.split('-')
            if len(parts) == 2:
                low = float(parts[0])
                high = float(parts[1])
                return (low + high) / 2  # Use average
        except:
            pass
    
    # Remove non-numeric characters except decimal point
    cleaned = re.sub(r'[^\d.]', '', elev_str)
    
    try:
        return float(cleaned) if cleaned else np.nan
    except:
        return np.nan

def clean_radon_concentration(radon_str):
    """Clean radon concentration values"""
    if pd.isna(radon_str):
        return np.nan
    
    radon_str = str(radon_str).strip()
    
    # Extract numeric part
    match = re.search(r'[\d.]+', radon_str)
    
    try:
        if match:
            return float(match.group())
        else:
            return np.nan
    except:
        return np.nan

def clean_coordinates(coord_str):
    """Clean coordinate strings"""
    if pd.isna(coord_str):
        return np.nan
    
    coord_str = str(coord_str).strip()
    
    # Extract numeric part
    match = re.search(r'[\d.]+', coord_str)
    
    try:
        if match:
            return float(match.group())
        else:
            return np.nan
    except:
        return np.nan

def parse_raw_data(raw_text):
    """
    Parse the raw data from the provided text format.
    This function handles the specific format of your data.
    """
    lines = raw_text.strip().split('\n')
    data_rows = []
    
    # Skip header
    for line in lines[1:]:
        if line.strip() == '':
            continue
        
        # Parse each line - the format appears to be:
        # Location | Radon | Year | X | Y | Rock | Soil | Elevation
        parts = re.split(r'\s{2,}', line.strip())  # Split by 2+ spaces
        
        if len(parts) >= 8:
            data_rows.append({
                'Location': parts[0].strip(),
                'Radon': parts[1].strip(),
                'Year': parts[2].strip(),
                'X': parts[3].strip(),
                'Y': parts[4].strip(),
                'Rock': parts[5].strip(),
                'Soil': parts[6].strip(),
                'Elevation': parts[7].strip()
            })
    
    return pd.DataFrame(data_rows)

def clean_radon_dataset(raw_data_path=None, raw_text=None):
    """
    Main function to clean radon dataset.
    Can accept either a file path or raw text data.
    """
    
    # Load data
    if raw_data_path and Path(raw_data_path).exists():
        # Read from file
        with open(raw_data_path, 'r') as f:
            raw_text = f.read()
    
    if raw_text is None:
        raise ValueError("Please provide either raw_data_path or raw_text")
    
    # Parse raw data
    df = parse_raw_data(raw_text)
    
    print(f"Initial rows: {len(df)}")
    print("\nOriginal data sample:")
    print(df.head())
    
    # CLEANING STEPS
    
    # 1. Clean numeric columns
    df['Radon'] = df['Radon'].apply(clean_radon_concentration)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['X'] = df['X'].apply(clean_coordinates)
    df['Y'] = df['Y'].apply(clean_coordinates)
    df['Elevation'] = df['Elevation'].apply(clean_elevation)
    
    # 2. Clean text columns
    df['Location'] = df['Location'].str.strip()
    df['Rock'] = df['Rock'].str.strip().replace('', np.nan)
    df['Soil'] = df['Soil'].str.strip().replace('', np.nan)
    
    # 3. Remove rows with missing critical values
    initial_rows = len(df)
    df = df.dropna(subset=['Radon', 'X', 'Y', 'Year'])
    removed = initial_rows - len(df)
    print(f"\nRows removed due to missing critical values: {removed}")
    
    # 4. Remove outliers (optional - radon > 100 KBq/m³ might be measurement errors)
    before_outlier_removal = len(df)
    df = df[df['Radon'] <= 100]  # Adjust threshold if needed
    print(f"Rows removed as outliers (Radon > 100): {before_outlier_removal - len(df)}")
    
    # 5. Fill missing elevations with median by location/region
    df['Elevation'].fillna(df['Elevation'].median(), inplace=True)
    
    # 6. Standardize rock and soil types
    rock_type_mapping = {
        'limestone': 'Limestone',
        'basalt': 'Basalt',
        'dolomite': 'Dolomite',
        'chalk': 'Chalk',
        'marl': 'Marl',
        'phosphate': 'Phosphate',
        'granite': 'Granite',
        'shale': 'Shale',
    }
    
    soil_type_mapping = {
        'terra rossa': 'Terra Rossa',
        'xerosols': 'Xerosols',
        'xerolls': 'Xerolls',
        'vertisols': 'Vertisols',
        'alluvial': 'Alluvial',
        'desert soil': 'Desert Soil',
        'clay': 'Clay',
    }
    
    # Apply mappings
    df['Rock'] = df['Rock'].fillna('Unknown')
    df['Soil'] = df['Soil'].fillna('Unknown')
    
    for old, new in rock_type_mapping.items():
        df['Rock'] = df['Rock'].str.lower().str.contains(old, na=False)
        if df['Rock'].any():
            df.loc[df['Rock'], 'Rock'] = new
    
    for old, new in soil_type_mapping.items():
        df['Soil'] = df['Soil'].str.lower().str.contains(old, na=False)
        if df['Soil'].any():
            df.loc[df['Soil'], 'Soil'] = new
    
    # 7. Add region classification
    def assign_region(x, y):
        if 35.45 <= x <= 35.55 and 32.28 <= y <= 32.38:
            return 'Irbid'
        elif 35.4 <= x <= 35.6 and 32.1 <= y <= 32.5:
            return 'Northern'
        elif 36.0 <= x <= 36.3 and 32.1 <= y <= 32.3:
            return 'Mafraq'
        elif 35.4 <= x <= 35.6 and 31.4 <= y <= 31.6:
            return 'Amman'
        else:
            return 'Other'
    
    df['Region'] = df.apply(lambda row: assign_region(row['X'], row['Y']), axis=1)
    
    # 8. Standardize coordinate system (ensure within reasonable bounds for Jordan)
    # Jordan coordinates roughly: 34.9-35.9 longitude, 29.2-32.5 latitude
    df = df[(df['X'] >= 34.9) & (df['X'] <= 35.9)]
    df = df[(df['Y'] >= 29.2) & (df['Y'] <= 32.5)]
    
    print(f"\nFinal cleaned dataset: {len(df)} rows")
    print("\nCleaned data sample:")
    print(df.head(10))
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nRegion distribution:")
    print(df['Region'].value_counts())
    
    print("\nRock type distribution:")
    print(df['Rock'].value_counts())
    
    print("\nSoil type distribution:")
    print(df['Soil'].value_counts())
    
    return df

def save_cleaned_data(df, output_path='radon_data_cleaned.csv'):
    """Save cleaned data to CSV"""
    df.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned data saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # RAW DATA - Copy your data here or load from file
    raw_data = """
Location                                        Average Radon Concetration(KBq/m^3)Year    X       Y       Rock                                Soil                   Elevation
historical city of Jerash                       2.292    2012  35.534 32.1612Limestone                           Terra Rossa                   600
Muwaqqar                                        6    2000 36.0738 31.4838Chalk Marl/Phosphatic               Yellow Desert Soil            900
Abu Nasier                                      1.3    2016 35.5254 32.0355Hard Limestone/Dolomite             Terra Rossa                  1000
Suwaileh                                        0.7    2016 35.4955 32.0058Limestone/Marl                      Deep Terra Rossa             1080
Al Rusaifah                                     2.5    2016 36.0147 32.0108Phosphorite/High uranium            Disturbed/Mining Tailin       650
Malka                                           1.53    2012 35.4707 32.4321chalky Limestone/Chert              Deep Terra Rossa              600
Al Rafid                                        3.39    2012 35.5052 32.4521chalky Limestone/Chert              Heavy Clay/Cracking soi       500
Aqraba                                          4.7    2012 35.2033 32.0739chalky Limestone/Chert              Terra Rossa                   450
Hubras                                          1.55    2011    35.5 32.3936chalky Limestone/Chert              Terra Rossa            550-600
Harta                                           1.03    2011 35.5046 32.4132chalky Limestone/Chert              Terra Rossa            500-550
Sabha                                           7.48    2016 36.2938 32.1956Basalt                              Xerolls                700-800
Um eljemal                                      5.83    2016 36.2138 32.1955Basalt                              Xerolls                700-800
Sama Alserhan                                   9.79    2016 36.1441  32.281Chalky Limestone                    Light colored dry clay 500-600
Mansourah - Mafraq                              4.09    2016 36.1014 32.2507Chalky Limestone                    Light colored dry clay 500-600
Al Manshia-Mafraq                               8.75    2016 36.0451 32.2213Limestone                           Desert soil                   700
Rehab-Mafraq                                    5.84    2016 36.0522  32.193Hard calcareous and flint           Terra Rossa/Calcium    800-900
Al Khaldiah-Mafraq                              8.70     2016 36.1853   32.11Church/Marl                         Desert soil                   700
Al Mafraq                                       4.21    2016 36.1321 32.2024Limestone                           Desert soil                   700
Balamah                                         7.75    2016 36.0511   32.14Hard calcareous and flint           Terra Rossa/Calcium    800-900
Alhamrah-Mafraq                                 7.15    2016 36.0916 32.2613Hard calcareous and flint           Terra Rossa/Calcium    800-900
Albweghdah                                      10.26    2016 36.0336 32.2809Hard calcareous and flint           Terra Rossa/Calcium    800-900
Housha                                          11.32    2016 36.0552 32.2706Hard calcareous and flint           Terra Rossa/Calcium    800-900
Muwaqqar                                        3.7    2008 36.0738 31.4838Chalk Marl/Phosphatic               Yellow Desert Soil            900
Wadi essir                                      4.1    2008 35.4639   31.56Wadi essir Limestone                Terra Rossa            600-950
Jabal Alakhdar-Jerash                           3    2012 35.5436 32.1716Limestone/Marl                      Terra Rossa                   700
Samma                                           7.4    2003 35.4127 32.3402Limestone                           Terra Rossa                   450
Al Rusaifah(abandoned phsphate mine             8.44    2002 36.0206 32.0117Phosphorite/High uranium            Disturbed/Mining Tailin       650
Amman                                           6.3    2002 35.5651 31.5008Limestone/Dolomite                  Terra Rossa            800-900
Irbid                                           4.09    2013  35.505 32.3309chalky Limestone/Chert              Terra Rossa                   600
Irbid                                           15.71    2013  35.505 32.3309chalky Limestone/Chert              Terra Rossa                   600
Irbid                                           24.19    2013  35.505 32.3309chalky Limestone/Chert              Terra Rossa                   600
Yarmouk River                                   0.48    2017 35.4303 32.4213Basalt/Limestone                    Alluvial                     -150
Al Fuhais                                       2.7    2003 35.4704 32.0014Limestone/Marl                      Terra Rossa                   900
Mazar Shamali                                   47.8    1999 35.4742 32.2756Hard Limestone                      Terra Rossa                   800
Kharja                                          1.32    2006 35.5311 32.3922Chalky Limestone                    Terra Rossa                   480
Housha                                          3.32    2008 36.0552 32.2706Hard calcareous and flint           Terra Rossa/Calcium    800-900
Housha                                          7.22    2008 36.0552 32.2706Hard calcareous and flint           Terra Rossa/Calcium    800-900
Housha                                          10.77    2008 36.0552 32.2706Hard calcareous and flint           Terra Rossa/Calcium    800-900
Housha                                          14.96    2008 36.0552 32.2706Hard calcareous and flint           Terra Rossa/Calcium    800-900
Housha                                          16.26    2008 36.0552 32.2706Hard calcareous and flint           Terra Rossa/Calcium    800-900
Soum                                            6.86    2003 35.4741 32.3513Hard Limestone                      Terra Rossa                   550
Al Rusaifah                                     7.28    2000 36.0147 32.0108Phosphorite/High uranium            Disturbed/Mining Tailin       650
Ayn jana                                        2.08    2012 35.4532 32.2024Limestone/Dolomite                  Terra Rossa            1000-1100
Umm Yanabe                                      2.36    2012 35.4556  32.222Limestone/Marl                      Terra Rossa            800-900
Ishtafena                                       3.62    2012 35.4653 32.2137Limestone/Dolomite                  Forest Soil            1000-1500
Al baag                                         1.9    2008 36.2023  32.221Limestone/Basalt                    Xerosols               650-700
Ajloun city                                     3.4    2008 35.4602 32.2029Limestone/Marl                      Terra Rossa            750-850
Al Hamra                                        15.15    2011 36.0916 32.2613Hard calcareous and flint           Terra Rossa/Calcium    800-900
Bayt Yafa                                       6.9    2017  35.472 32.3108Chalky Limestone                    Terra Rossa            550-650
Al Dajania-mafraq                               32.9    2009 36.0239  32.202Limestone/Marl/Chert                Silty-Clay                    600
Al Dajania-mafraq                               8.5    2009 36.0239  32.202Limestone/Marl/Chert                Silty-Clay                    600
Al Dajania-mafraq                               2.55    2009 36.0239  32.202Limestone/Marl/Chert                Silty-Clay                    600
Soum-irbid                                      7.7    2017 35.4742 32.3513Chalky Limestone                    Terra Rossa                   500
Just                                            0.056    1998 35.5928 32.2942Limestone                           Vertisols              580-610
Ramtha                                          0.062    1998  36.001 32.3227Limestone                           Vertisols              500-530
Torra                                           0.062    1998  35.592 32.3805Chalky Marl                         Marly Soil             450-480
Shajera                                         0.063    1998 35.5623 32.3836Chalky Marl                         Deep cultivation radish460-490
Emrawa                                          0.065    1998 35.5608  32.405Chalky Limestone                    Shallow Soil           350-400
Ethnaibeh                                       0.065    1998 35.5444 32.4128Chalky Limestone                    Eroded Soil            300-350
Eidon                                           4.62    2004 35.5113 32.3125Chalky Limestone                    Terra Rossa/Heavy Clay 600-720
Eidon                                           9.35    2004 35.5113 32.3225Chalky Limestone                    Terra Rossa/Heavy Clay 600-720
Attayba                                         0.051    2021 35.4259 32.3234Turonian Limestone                  Terra Rossa            350-500
Mazar shamali                                   0.02    2021 35.4742 32.2756Turonian Limestone                  Terra Rossa            750-850
Anjarah                                         3.94    2017 35.4527 32.1818Limestone/Marl                      Terra Rossa/Shallow    850-950
Ajloun city                                     3.63    2017 35.4602 32.2029Limestone/Marl                      Terra Rossa            750-850
Kufranjah                                       4.22    2017 35.4209 32.1743Limestone/Marl                      Alluvial Soils                400
Ayn jana                                        4.01    2017 35.4532 32.2024Limestone/Dolomite                  Terra Rossa            1000-1100
Ibbin                                           4.84    2017 35.4845 32.2135Dolomitic Limestone                 Terra Rossa            1050-1100
Ballas                                          3.21    2017  35.421 32.1545Limestone/Shale                     Terra Rossa                   700
Ayn Albustan                                    4.19    2017 35.4314 32.1815Limestone/Marl                      Alluvial Terra Rossa          650
Mafraq                                          0.05    2004 36.1208 32.2028Limestone                           Desert soil                   700
jerash                                          0.048    2004 35.5344  32.162Limestone                           Terra Rossa                   600
Ajlun                                           0.04    2004 35.4505 32.1954Limestone/Marl                      Terra Rossa            750-850
Madaba                                          0.092    2004 35.4735 31.4303Limestone/Marl                      Terra Rossa            750-820
Salt                                            0.046    2004 35.4335 32.0155Limestone/Marl                      Terra Rossa            700-1100
Tafila                                          0.047    2004 35.3656 30.4954Limestone/Phosphate/Basalt          Terra Rossa            400-1000
Karak                                           0.099    2004 35.4216 31.1054Limestone/Phosphate/Marl            Terra Rossa/Xerosols   800-1100
Ma'an                                           0.096    2004 35.4324 30.1058Dolomitic Limestone/Graniet/PhosphatYermosols/Aridisols    1000-1500
Aqaba                                           0.029    2004 35.0029 29.3152Graniet                             Granitic Sands               1000
Mutah-Karak                                     0.12    2015 35.4146 31.0532Dolomite Limestone                  Terra Rossa                  1100
Madden-Karak                                    0.1    2015 35.4357 31.0659Chalky Marl                         Terra Rossa                  1050
Rakeen-Karak                                    0.327    2015 35.4218 31.1329Limestone                           Terra Rossa                  1000
Barada-Karak                                    0.356    2015 35.4032 31.1243Graniet                             Dry sand                     1000
Adder-Karak                                     0.22    2015 35.4533  31.122Limestone/Phosphate                 Terra Rossa                   950
Al-Haweya-Karak                                 0.11    2015 35.4349 31.0101Limestone/Clay                      Terra Rossa                   920
Al-smakia-Karak                                 0.355    2015 35.4755 31.1816limestone/Graniet                   Terra Rossa                   980
Ma'an - Shoubak                                 4.037    1998 35.3325 30.3103Limestone/Phosphate                 Yellowish Brown Thin         1400
"""
    
    # Clean the dataset
    df_cleaned = clean_radon_dataset(raw_text=raw_data)
    
    # Save to CSV
    csv_path = save_cleaned_data(df_cleaned)
    
    print("\n" + "="*60)
    print("DATA CLEANING COMPLETE!")
    print("="*60)
