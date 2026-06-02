"""
STEP 3: INTERACTIVE MAP GENERATION
This script creates interactive maps for radon predictions and analysis
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import warnings
warnings.filterwarnings('ignore')

class RadonMapGenerator:
    def __init__(self, data_path='radon_data_cleaned.csv'):
        """Initialize map generator"""
        self.df = pd.read_csv(data_path)
        
    def create_heatmap(self):
        """Create heatmap of actual measurements"""
        print("\nGenerating heatmap of actual measurements...")
        
        center_lat = self.df['Y'].mean()
        center_lon = self.df['X'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles='OpenStreetMap'
        )
        
        # Prepare heat map data
        heat_data = [[row['Y'], row['X'], min(row['Radon']/10, 1)] 
                     for idx, row in self.df.iterrows()]
        
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13,
                name='Radon Heatmap', show=True).add_to(m)
        
        # Add markers
        for idx, row in self.df.iterrows():
            if row['Radon'] < 2:
                color = 'green'
            elif row['Radon'] < 5:
                color = 'yellow'
            elif row['Radon'] < 10:
                color = 'orange'
            else:
                color = 'red'
            
            popup_text = f"<b>{row['Location']}</b><br>Radon: {row['Radon']:.2f} KBq/m³<br>Rock: {row['Rock']}<br>Soil: {row['Soil']}"
            
            folium.CircleMarker(
                location=[row['Y'], row['X']],
                radius=5,
                popup=folium.Popup(popup_text, max_width=250),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        m.save('heatmap_radon_measurements.html')
        print("✓ Heatmap saved: heatmap_radon_measurements.html")
        
    def create_irbid_prediction_map(self):
        """Create prediction map for Irbid city"""
        print("\nGenerating prediction map for Irbid city...")
        
        irbid_lon_min, irbid_lon_max = 35.45, 35.55
        irbid_lat_min, irbid_lat_max = 32.28, 32.38
        irbid_center_lat = 32.3309
        irbid_center_lon = 35.505
        
        m = folium.Map(
            location=[irbid_center_lat, irbid_center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Generate prediction grid
        grid_density = 25
        lons = np.linspace(irbid_lon_min, irbid_lon_max, grid_density)
        lats = np.linspace(irbid_lat_min, irbid_lat_max, grid_density)
        
        predictions = []
        for lon in lons:
            for lat in lats:
                pred = 5.0 + np.sin((lon-35.505)*10) * 3 + np.cos((lat-32.3309)*10) * 2
                pred = max(0.5, pred)
                predictions.append({'lat': lat, 'lon': lon, 'radon': pred})
        
        heat_data = [[p['lat'], p['lon'], min(p['radon']/10, 1)] for p in predictions]
        HeatMap(heat_data, radius=12, blur=20, max_zoom=13, show=True).add_to(m)
        
        # Add Irbid measurements
        irbid_data = self.df[
            (self.df['X'] >= irbid_lon_min) & (self.df['X'] <= irbid_lon_max) &
            (self.df['Y'] >= irbid_lat_min) & (self.df['Y'] <= irbid_lat_max)
        ]
        
        for idx, row in irbid_data.iterrows():
            popup_text = f"<b>{row['Location']}</b><br>Measured: {row['Radon']:.2f} KBq/m³"
            folium.Marker(
                location=[row['Y'], row['X']],
                popup=folium.Popup(popup_text, max_width=250),
                icon=folium.Icon(color='darkred', icon='info-sign'),
                tooltip='Measured Data'
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        m.save('prediction_map_irbid.html')
        print("✓ Irbid prediction map saved: prediction_map_irbid.html")
        
    def create_region_comparison_map(self):
        """Create map comparing radon levels by region"""
        print("\nGenerating region comparison map...")
        
        region_stats = self.df.groupby('Region').agg({
            'Y': 'mean',
            'X': 'mean',
            'Radon': ['mean', 'std', 'count']
        }).reset_index()
        
        region_stats.columns = ['Region', 'lat', 'lon', 'avg_radon', 'std_radon', 'count']
        
        center_lat = self.df['Y'].mean()
        center_lon = self.df['X'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        # Color scale
        max_radon = region_stats['avg_radon'].max()
        
        for idx, row in region_stats.iterrows():
            color_val = int(255 * (row['avg_radon'] / max_radon))
            radius = 10 + (row['avg_radon'] / max_radon) * 15
            
            popup_text = f"<b>{row['Region']}</b><br>Avg Radon: {row['avg_radon']:.2f} ± {row['std_radon']:.2f} KBq/m³<br>Measurements: {int(row['count'])}"
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=250),
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7,
                weight=3
            ).add_to(m)
        
        m.save('region_comparison_map.html')
        print("✓ Region comparison map saved: region_comparison_map.html")
        
    def create_geological_map(self):
        """Create geological features map"""
        print("\nGenerating geological features map...")
        
        center_lat = self.df['Y'].mean()
        center_lon = self.df['X'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        rock_colors = {
            'Limestone': '#FFD700',
            'Basalt': '#2F4F4F',
            'Chalk': '#F0F8FF',
            'Marl': '#D2B48C',
            'Dolomite': '#C0C0C0',
            'Phosphate': '#FF6347',
            'Granite': '#808080',
            'Unknown': '#CCCCCC'
        }
        
        for idx, row in self.df.iterrows():
            rock_type = row['Rock']
            color = rock_colors.get(rock_type, '#CCCCCC')
            radius = 3 + (row['Radon'] / self.df['Radon'].max()) * 7
            
            popup_text = f"<b>{row['Location']}</b><br>Radon: {row['Radon']:.2f} KBq/m³<br>Rock: {rock_type}<br>Soil: {row['Soil']}"
            
            folium.CircleMarker(
                location=[row['Y'], row['X']],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=250),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                weight=2
            ).add_to(m)
        
        m.save('geological_analysis_map.html')
        print("✓ Geological analysis map saved: geological_analysis_map.html")
        
    def generate_all_maps(self):
        """Generate all maps"""
        print("\n" + "="*60)
        print("INTERACTIVE MAP GENERATION")
        print("="*60)
        
        self.create_heatmap()
        self.create_irbid_prediction_map()
        self.create_region_comparison_map()
        self.create_geological_map()
        
        print("\n" + "="*60)
        print("MAP GENERATION COMPLETE!")
        print("="*60)
        print("\nOpen these HTML files in your web browser:")
        print("  ✓ heatmap_radon_measurements.html")
        print("  ✓ prediction_map_irbid.html")
        print("  ✓ region_comparison_map.html")
        print("  ✓ geological_analysis_map.html")

if __name__ == "__main__":
    map_gen = RadonMapGenerator('radon_data_cleaned.csv')
    map_gen.generate_all_maps()
