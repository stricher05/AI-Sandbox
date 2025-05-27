"""
Generate sample data for the ML Forecasting Platform
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.preprocessing import generate_sample_data

if __name__ == "__main__":
    print("🔢 Generating sample energy demand data...")
    generate_sample_data("data/sample_energy_data.csv", n_samples=8760)
    print("✅ Sample data generated successfully!")
    print("📊 Dataset contains 8760 hourly records (1 year)")
    print("📁 File saved to: data/sample_energy_data.csv")
