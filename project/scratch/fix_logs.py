import pandas as pd
import os

csv_path = '/Users/HetviSheth/Flood_Prediction/project/logs/monitor_log.csv'
if os.path.exists(csv_path):
    # Read the existing CSV
    df = pd.read_csv(csv_path)
    print(f"Original Columns: {df.columns.tolist()}")
    
    # Identify if it's the 'messed up' version where Rain is in Animals column, etc.
    # We can detect this by checking if 'Rain' column contains string risk levels like 'HIGH'
    if 'Rain' in df.columns and df['Rain'].dtype == object:
        print("Detected swapped columns. Fixing...")
        
        # Mapping for the swapped columns we found:
        # 1. Timestamp
        # 2. Water % -> water_p
        # 3. Objects -> obj_summary
        # 4. Animals -> rain (should be col 5)
        # 5. Rain -> risk_level (should be col 6)
        # 6. Risk -> risk_score (should be col 7?)
        
        # Let's just re-align based on the 7-column append logic we saw
        # Row was: [Timestamp, water_p, obj_summary, rain, risk_level, risk_score, location]
        # Header was: [Timestamp, Water %, Objects, Animals, Rain, Risk, Location]
        
        fixed_df = df.copy()
        fixed_df['Location'] = df['Location']
        fixed_df['Risk_Score'] = df['Risk'] # Risk (index 5) was getting risk_score
        fixed_df['Risk_Level'] = df['Rain'] # Rain (index 4) was getting risk_level
        fixed_df['Rain_mm'] = df['Animals'] # Animals (index 3) was getting rain
        fixed_df['Object_Summary'] = df['Objects']
        fixed_df['Water_P'] = df['Water %']
        
        # Drop old columns
        new_df = fixed_df[['Timestamp', 'Water_P', 'Object_Summary', 'Rain_mm', 'Risk_Level', 'Risk_Score', 'Location']]
        
        # Save fixed version
        new_df.to_csv(csv_path, index=False)
        print("✅ monitor_log.csv fixed successfully.")
    else:
        print("CSV seems to have correct types or is already fixed.")
