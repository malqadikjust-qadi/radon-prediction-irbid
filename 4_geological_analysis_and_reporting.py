"""
STEP 4: COMPREHENSIVE GEOLOGICAL ANALYSIS AND REPORTING
This script performs deep geological and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class GeologicalAnalyzer:
    def __init__(self, data_path='radon_data_cleaned.csv'):
        """Initialize geological analyzer"""
        self.df = pd.read_csv(data_path)
        
    def correlation_analysis(self):
        """Analyze correlations between variables"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        numeric_cols = ['Radon', 'X', 'Y', 'Elevation', 'Year']
        df_numeric = self.df[numeric_cols].copy()
        
        corr_matrix = df_numeric.corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.3f', ax=ax)
        ax.set_title('Correlation Matrix: Radon and Geographic Features')
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Correlation heatmap saved: correlation_analysis.png")
        plt.close()
        
    def rock_soil_analysis(self):
        """Detailed analysis of rock and soil types"""
        print("\n" + "="*60)
        print("ROCK AND SOIL TYPE ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        rock_analysis = self.df.groupby('Rock')['Radon'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        
        print("\nRock Type Statistics:")
        print(rock_analysis)
        
        axes[0, 0].barh(range(len(rock_analysis)), rock_analysis['mean'].values, 
                       xerr=rock_analysis['std'].values, capsize=5, alpha=0.7, color='skyblue')
        axes[0, 0].set_yticks(range(len(rock_analysis)))
        axes[0, 0].set_yticklabels(rock_analysis.index, fontsize=9)
        axes[0, 0].set_xlabel('Average Radon (KBq/m³)')
        axes[0, 0].set_title('Rock Type Impact on Radon Levels')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        axes[0, 1].bar(range(len(rock_analysis)), rock_analysis['count'].values, alpha=0.7, color='coral')
        axes[0, 1].set_xticks(range(len(rock_analysis)))
        axes[0, 1].set_xticklabels(rock_analysis.index, rotation=45, ha='right', fontsize=9)
        axes[0, 1].set_ylabel('Number of Measurements')
        axes[0, 1].set_title('Sample Size by Rock Type')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        soil_analysis = self.df.groupby('Soil')['Radon'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        
        print("\nSoil Type Statistics:")
        print(soil_analysis)
        
        axes[1, 0].barh(range(len(soil_analysis)), soil_analysis['mean'].values,
                       xerr=soil_analysis['std'].values, capsize=5, alpha=0.7, color='lightgreen')
        axes[1, 0].set_yticks(range(len(soil_analysis)))
        axes[1, 0].set_yticklabels(soil_analysis.index, fontsize=9)
        axes[1, 0].set_xlabel('Average Radon (KBq/m³)')
        axes[1, 0].set_title('Soil Type Impact on Radon Levels')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Regional comparison
        region_stats = self.df.groupby('Region')['Radon'].mean().sort_values(ascending=False)
        axes[1, 1].bar(region_stats.index, region_stats.values, alpha=0.7, color='purple')
        axes[1, 1].set_ylabel('Average Radon (KBq/m³)')
        axes[1, 1].set_title('Average Radon by Region')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('rock_soil_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Rock-soil analysis plot saved: rock_soil_analysis.png")
        plt.close()
        
    def regional_statistical_analysis(self):
        """Statistical analysis by region"""
        print("\n" + "="*60)
        print("REGIONAL STATISTICAL ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        region_stats = self.df.groupby('Region')['Radon'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        
        print("\nRegional Statistics:")
        print(region_stats)
        
        self.df.boxplot(column='Radon', by='Region', ax=axes[0, 0])
        axes[0, 0].set_title('Radon Distribution by Region')
        axes[0, 0].set_xlabel('Region')
        axes[0, 0].set_ylabel('Radon (KBq/m³)')
        
        region_order = region_stats.index.tolist()
        sns.violinplot(data=self.df, x='Region', y='Radon', order=region_order, ax=axes[0, 1])
        axes[0, 1].set_title('Radon Distribution (Violin Plot)')
        
        axes[1, 0].bar(range(len(region_stats)), region_stats['mean'].values,
                      yerr=region_stats['std'].values, capsize=5, alpha=0.7, color='purple')
        axes[1, 0].set_xticks(range(len(region_stats)))
        axes[1, 0].set_xticklabels(region_stats.index, rotation=45)
        axes[1, 0].set_ylabel('Average Radon (KBq/m³)')
        axes[1, 0].set_title('Mean Radon by Region')
        
        region_counts = self.df['Region'].value_counts()
        axes[1, 1].pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Measurements by Region')
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('regional_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Regional analysis plot saved: regional_analysis.png")
        plt.close()
        
        # ANOVA test
        print("\n" + "-"*40)
        print("ANOVA Test (Region Effect):")
        groups = [self.df[self.df['Region'] == region]['Radon'].values 
                 for region in self.df['Region'].unique()]
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.6f}")
        
    def elevation_analysis(self):
        """Analyze elevation vs radon"""
        print("\n" + "="*60)
        print("ELEVATION ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(self.df['Elevation'], self.df['Radon'], alpha=0.6, s=50, edgecolors='k')
        z = np.polyfit(self.df['Elevation'], self.df['Radon'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.df['Elevation'].min(), self.df['Elevation'].max(), 100)
        axes[0].plot(x_trend, p(x_trend), "r--", linewidth=2)
        axes[0].set_xlabel('Elevation (m)')
        axes[0].set_ylabel('Radon (KBq/m³)')
        axes[0].set_title('Radon vs Elevation')
        axes[0].grid(True, alpha=0.3)
        
        elevation_bins = pd.cut(self.df['Elevation'], bins=5)
        elev_bin_stats = self.df.groupby(elevation_bins)['Radon'].agg(['mean', 'std', 'count'])
        bin_centers = [interval.mid for interval in elev_bin_stats.index]
        
        axes[1].errorbar(bin_centers, elev_bin_stats['mean'], 
                        yerr=elev_bin_stats['std'], fmt='o-', capsize=5,
                        markersize=8, linewidth=2)
        axes[1].set_xlabel('Elevation (m)')
        axes[1].set_ylabel('Average Radon (KBq/m³)')
        axes[1].set_title('Average Radon by Elevation Range')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('elevation_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Elevation analysis plot saved: elevation_analysis.png")
        plt.close()
        
    def temporal_analysis(self):
        """Analyze radon trends over time"""
        print("\n" + "="*60)
        print("TEMPORAL ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        yearly_stats = self.df.groupby('Year')['Radon'].agg(['mean', 'std', 'count', 'median']).reset_index()
        
        print("\nYearly Statistics:")
        print(yearly_stats)
        
        axes[0, 0].plot(yearly_stats['Year'], yearly_stats['mean'], marker='o', 
                       linewidth=2, markersize=8, label='Mean', color='blue')
        axes[0, 0].fill_between(yearly_stats['Year'], 
                               yearly_stats['mean'] - yearly_stats['std'],
                               yearly_stats['mean'] + yearly_stats['std'],
                               alpha=0.3, color='blue')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Radon (KBq/m³)')
        axes[0, 0].set_title('Temporal Trend: Radon Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        self.df.boxplot(column='Radon', by='Year', ax=axes[0, 1])
        axes[0, 1].set_title('Radon Distribution by Year')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Radon (KBq/m³)')
        
        axes[1, 0].bar(yearly_stats['Year'], yearly_stats['count'], alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Number of Measurements')
        axes[1, 0].set_title('Sample Size by Year')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        decade_bins = pd.cut(self.df['Year'], bins=[1990, 2000, 2010, 2020, 2030], 
                            labels=['1990s', '2000s', '2010s', '2020s'], include_lowest=True)
        decade_data = self.df.groupby(decade_bins)['Radon'].apply(list)
        axes[1, 1].boxplot([decade_data[d] for d in decade_data.index if pd.notna(d)],
                          labels=[d for d in decade_data.index if pd.notna(d)])
        axes[1, 1].set_ylabel('Radon (KBq/m³)')
        axes[1, 1].set_title('Radon Distribution by Decade')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Temporal analysis plot saved: temporal_analysis.png")
        plt.close()
        
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*80)
        print(" "*20 + "COMPREHENSIVE GEOLOGICAL ANALYSIS REPORT")
        print("="*80)
        
        print("\nDATASET OVERVIEW")
        print("-"*60)
        print(f"Total measurements: {len(self.df)}")
        print(f"Regions: {self.df['Region'].nunique()}")
        print(f"Rock types: {self.df['Rock'].nunique()}")
        print(f"Year range: {int(self.df['Year'].min())} - {int(self.df['Year'].max())}")
        print(f"Radon range: {self.df['Radon'].min():.2f} - {self.df['Radon'].max():.2f} KBq/m³")
        print(f"Mean radon: {self.df['Radon'].mean():.2f} ± {self.df['Radon'].std():.2f} KBq/m³")
        
        self.correlation_analysis()
        self.rock_soil_analysis()
        self.regional_statistical_analysis()
        self.elevation_analysis()
        self.temporal_analysis()
        
        print("\n" + "="*60)
        print("TOP 10 HIGH-RADON LOCATIONS")
        print("="*60)
        top_radon = self.df.nlargest(10, 'Radon')[['Location', 'Radon', 'Rock', 'Region']]
        for idx, row in top_radon.iterrows():
            print(f"{row['Location']:30s} | {row['Radon']:7.2f} KBq/m³ | {row['Rock']:20s} | {row['Region']}")
        
        print("\n" + "="*80)
        print(" "*30 + "ANALYSIS COMPLETE!")
        print("="*80)

if __name__ == "__main__":
    analyzer = GeologicalAnalyzer('radon_data_cleaned.csv')
    analyzer.generate_report()
