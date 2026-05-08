# project/src/utils/risk.py

# Standardized Risk Levels
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
DANGEROUS = "DANGEROUS"

def get_flood_risk(water_p, has_person=False, has_animal=False, has_vehicle=False, has_crocodile=False, oif_detected=False):
    """
    Production-level Risk Engine with Crocodile Priority.
    Crocodiles in flood water are extremely dangerous and get highest priority.
    Returns: (risk_level, risk_score)
    """
    score = 0
    
    # 1. Base Score (Max 60 pts) - Increased water sensitivity
    score += min(60, water_p * 0.6)
    
    # 2. Special Rule: Very high water coverage automatically dangerous
    if water_p > 80:
        return DANGEROUS, 100
    elif water_p > 60:
        score += 20  # Bonus for high water coverage
    
    # 3. Object Multipliers (Crocodile & Animal Priority)
    if has_crocodile: 
        score += 30  # Crocodiles are extremely dangerous in floods
    if has_person: score += 15
    if has_animal: score += 35 # [VIVA FIX] Jump straight to high priority
    if has_vehicle: score += 5
    
    # 4. Object-in-Flood (OIF)
    if oif_detected:
        if has_crocodile: score += 35  # Crocodile in water = immediate danger
        elif has_person: score += 25
        elif has_animal: score += 40 # Animal in water = HIGH RISK
        else: score += 10
    
    # 5. Crocodile/Animal Special Rule: Forced High Risk
    if has_animal and water_p > 2:
        return HIGH, max(70, int(score))
    if has_crocodile and water_p > 5:
        return DANGEROUS, 100
    
    score = min(100, score)
    # 6. Categorization
    if score < 15: level = LOW
    elif score < 40: level = MEDIUM
    elif score < 70: level = HIGH
    else: level = DANGEROUS
    return level, int(score)
