import torch
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter
import random

class FloodAugmentations:
    """
    Simulation of physical flood effects for data augmentation.
    """
    
    @staticmethod
    def add_water_reflection(image, mask):
        """
        Simulate mirror-like reflections on water surfaces.
        """
        img_np = np.array(image)
        # Identify water areas from mask (assuming water classes are 0, 1, 2)
        water_mask = (mask < 3).astype(np.uint8)
        
        # Create a flipped version of the non-water part of the image
        flipped = cv2.flip(img_np, 0) # Flip vertically
        
        # Blend flipped image into water areas
        alpha = 0.3 # Reflection intensity
        res = img_np.copy()
        res[water_mask == 1] = cv2.addWeighted(img_np[water_mask == 1], 1 - alpha, 
                                              flipped[water_mask == 1], alpha, 0)
        return Image.fromarray(res)

    @staticmethod
    def simulate_water_turbidity(image, mask):
        """
        Add murkiness/brown tint to water areas.
        """
        img_np = np.array(image).astype(np.float32)
        water_mask = (mask < 3).astype(np.uint8)[:, :, np.newaxis]
        
        # Brownish silt tint
        silt_color = np.array([60, 80, 100]) # BGR-ish
        
        # Apply tint and a bit of blur to water areas
        blurred = cv2.GaussianBlur(img_np, (5, 5), 2)
        tinted = 0.7 * img_np + 0.3 * silt_color
        
        res = img_np * (1 - water_mask) + (tinted * 0.8 + blurred * 0.2) * water_mask
        return Image.fromarray(res.astype(np.uint8))

    @staticmethod
    def simulate_partial_submersion(image, mask):
        """
        Simulate objects (cars, people) being partially covered by water.
        """
        # This is complex to do purely on pixels without 3D, 
        # but we can simulate a 'water line' gradient.
        img_np = np.array(image)
        h, w, _ = img_np.shape
        water_line = random.randint(h // 2, h)
        
        # Apply a semi-transparent 'flood' layer below the water line
        flood_layer = np.full_like(img_np, (40, 50, 70)) # Murky water color
        mask_layer = np.zeros((h, w, 1), dtype=np.uint8)
        mask_layer[water_line:, :] = 1
        
        res = cv2.addWeighted(img_np, 0.7, flood_layer, 0.3, 0)
        final = img_np.copy()
        final[water_line:, :] = res[water_line:, :]
        
        # Update mask to reflect new water area
        new_mask = mask.copy()
        new_mask[water_line:, :] = 0 # flood_water class
        
        return Image.fromarray(final), new_mask

    @staticmethod
    def add_rain_effect(image):
        """
        Add synthetic rain streaks.
        """
        img_np = np.array(image)
        rain_layer = np.zeros_like(img_np)
        
        for _ in range(500):
            x = random.randint(0, img_np.shape[1] - 1)
            y = random.randint(0, img_np.shape[0] - 1)
            length = random.randint(5, 20)
            cv2.line(rain_layer, (x, y), (x, y + length), (200, 200, 200), 1)
            
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        res = cv2.addWeighted(img_np, 1.0, rain_layer, 0.2, 0)
        return Image.fromarray(res)

def apply_flood_augmentation_pipeline(image, mask):
    """
    Full pipeline wrapper
    """
    p = FloodAugmentations()
    
    # 30% chance for reflection
    if random.random() < 0.3:
        image = p.add_water_reflection(image, mask)
    
    # 40% chance for turbidity
    if random.random() < 0.4:
        image = p.simulate_water_turbidity(image, mask)
    
    # 30% chance for submersion
    if random.random() < 0.3:
        image, mask = p.simulate_partial_submersion(image, mask)
    
    # 50% chance for rain
    if random.random() < 0.5:
        image = p.add_rain_effect(image)
        
    return image, mask
