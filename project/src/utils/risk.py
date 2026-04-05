# project/src/utils/risk.py

# Standardized Risk Levels
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
DANGEROUS = "DANGEROUS"

def get_flood_risk(water_p, has_person=False, has_animal=False, has_vehicle=False, oif_detected=False):
    """
    Production-level Risk Engine.
    Returns: (risk_level, risk_score)
    """
    score = 0
    # 1. Base Score (Max 50 pts)
    score += min(50, water_p * 0.5)
    # 2. Object Multipliers
    if has_person: score += 15
    if has_animal: score += 10
    if has_vehicle: score += 5
    # 3. Object-in-Flood (OIF)
    if oif_detected:
        if has_person: score += 25
        elif has_animal: score += 15
        else: score += 10
    score = min(100, score)
    # 4. Categorization
    if score < 15: level = LOW
    elif score < 40: level = MEDIUM
    elif score < 70: level = HIGH
    else: level = DANGEROUS
    return level, int(score)
