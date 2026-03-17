import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import os
import json
from model_dfs import DynamicFocusNet
from utils_dfs import visualize_attention

class DynamicFocusPredictor:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = DynamicFocusNet(num_classes=3, pretrained=False)
        # Load weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model path {model_path} not found. Using random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['COVID-19', 'Non-COVID', 'Normal']

    def predict_single(self, image_path, save_vis_path=None):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Preprocessing (similar to training but without mask for now unless provided)
        # Assuming inference on raw images. If lung masks are available, they should be applied.
        # Here we just resize to 512x512
        img_resized = cv2.resize(img, (512, 512))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        # Get predictions
        probs = torch.softmax(outputs['pred_fused'], dim=1)
        score, pred_idx = torch.max(probs, 1)
        pred_class = self.classes[pred_idx.item()]
        
        # Get attention and crop
        att_map = outputs['att_map']
        crop_coords = outputs['crop_coords'][0] # (x1, y1, x2, y2)
        
        result = {
            'class': pred_class,
            'confidence': score.item(),
            'probabilities': {cls: prob.item() for cls, prob in zip(self.classes, probs[0])},
            'crop_coords': crop_coords
        }
        
        if save_vis_path:
            # Visualize
            # Use the normalized tensor for visualization helper
            visualize_attention(img_tensor[0], att_map[0], crop_coords, save_vis_path)
            
        return result

    def predict_batch(self, image_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for img_name in os.listdir(image_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(image_dir, img_name)
            vis_path = os.path.join(output_dir, f"vis_{img_name}")
            
            try:
                res = self.predict_single(img_path, vis_path)
                res['image_id'] = img_name
                results.append(res)
                print(f"Processed {img_name}: {res['class']} ({res['confidence']:.4f})")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                
        # Save results json
        with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        return results

if __name__ == '__main__':
    # Example usage
    # Assuming we have a trained model (which we don't yet, so this will warn)
    predictor = DynamicFocusPredictor('/Users/duanhao/Desktop/Fine-grained-recognition-of-medical-images-based-on-dynamic-focus-search/ablation_outputs/dfs_full/fold_5/best_model.pth')
    
    # Test on a few images from dataset if available
    test_dir = '/Users/duanhao/Desktop/Fine-grained-recognition-of-medical-images-based-on-dynamic-focus-search/dataset/Test/COVID-19/images'
    if os.path.exists(test_dir):
        print(f"Running inference on {test_dir}...")
        predictor.predict_batch(test_dir, 'inference_results')
    else:
        print("Test directory not found.")
