"""
Confidence-based routing system for landmark detection.
Routes predictions through CV model -> LLaVA fallback based on confidence.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from PIL import Image

from models.landmark_detector import LandmarkDetector


class ConfidenceRouter:
    """
    Routes landmark predictions with user confirmation feedback.
    
    Two modes:
    1. Auto mode (confidence thresholds):
       - High confidence (>80%): Return CV prediction
       - Medium confidence (60-80%): Return CV + suggest LLaVA
       - Low confidence (<60%): Suggest LLaVA directly
    
    2. Confirmation mode (human-in-the-loop):
       - Show CV prediction with top-5 options
       - User confirms or rejects
       - If rejected: fallback to LLaVA
       - Learns from user feedback over time
    """
    
    def __init__(
        self,
        model_path: str,
        class_mapping_path: str,
        high_confidence_threshold: float = 0.80,
        low_confidence_threshold: float = 0.60,
        device: str = None,
        feedback_log_path: Optional[str] = None
    ):
        """
        Initialize the confidence router.
        
        Args:
            model_path: Path to trained model checkpoint
            class_mapping_path: Path to class_mapping.json
            high_confidence_threshold: Threshold for auto mode (default 80%)
            low_confidence_threshold: Threshold for auto mode (default 60%)
            device: Device to run model on ('cuda' or 'cpu')
            feedback_log_path: Path to save user feedback (optional)
        """
        self.high_threshold = high_confidence_threshold
        self.low_threshold = low_confidence_threshold
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        num_classes = self.class_mapping['num_classes']
        
        # Load model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LandmarkDetector(num_classes=num_classes)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.model.load_state_dict(checkpoint)
        
        self.model.model.to(self.device)
        self.model.model.eval()
        
        # Create reverse mapping (idx -> landmark name)
        self.idx_to_landmark = {
            v: k for k, v in self.class_mapping['landmark_to_idx'].items()
        }
        
        # User feedback tracking
        self.feedback_log_path = feedback_log_path
        self.feedback_history = []
        
        print(f"✓ Loaded {num_classes}-class model")
        print(f"✓ High confidence threshold: {high_confidence_threshold*100:.0f}%")
        print(f"✓ Low confidence threshold: {low_confidence_threshold*100:.0f}%")
        if feedback_log_path:
            print(f"✓ Feedback logging enabled: {feedback_log_path}")
    
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        mode: str = 'confirmation'
    ) -> Dict:
        """
        Predict landmark with routing logic.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            mode: 'confirmation' (wait for user) or 'auto' (threshold-based)
            
        Returns:
            Dictionary with routing decision and predictions:
            {
                'action': 'show_confirmation' | 'return_cv' | 'verify_with_llava' | 'use_llava',
                'confidence': float,
                'top_prediction': str,
                'top_k_predictions': [(landmark, confidence), ...],
                'needs_user_confirmation': bool,  # For confirmation mode
                'needs_llava': bool,  # For auto mode
                'recommendation': str  # UI hint for user
            }
        """
        # Get CV model predictions
        predictions = self.model.predict(image, top_k=top_k)
        
        # Extract top prediction and confidence
        top_pred = predictions[0]
        top_class_idx = top_pred['class_idx']
        top_confidence = top_pred['confidence']
        top_landmark = self.idx_to_landmark[top_class_idx]
        
        # Build top-k list
        top_k_list = [
            (self.idx_to_landmark[p['class_idx']], p['confidence'])
            for p in predictions
        ]
        
        # Routing decision based on mode
        if mode == 'confirmation':
            # Confirmation mode: Always show CV prediction + confirmation UI
            return {
                'action': 'show_confirmation',
                'confidence': top_confidence,
                'top_prediction': top_landmark,
                'top_k_predictions': top_k_list,
                'needs_user_confirmation': True,
                'recommendation': self._get_confidence_hint(top_confidence)
            }
        else:
            # Auto mode: Threshold-based routing
            if top_confidence >= self.high_threshold:
                action = 'return_cv'
                needs_llava = False
            elif top_confidence >= self.low_threshold:
                action = 'verify_with_llava'
                needs_llava = True
            else:
                action = 'use_llava'
                needs_llava = True
            
            return {
                'action': action,
                'confidence': top_confidence,
                'top_prediction': top_landmark,
                'top_k_predictions': top_k_list,
                'needs_llava': needs_llava
            }
    
    def _get_confidence_hint(self, confidence: float) -> str:
        """Get UI recommendation based on confidence."""
        if confidence >= 0.90:
            return "Very confident"
        elif confidence >= 0.75:
            return "Confident"
        elif confidence >= 0.60:
            return "Uncertain - consider checking"
        else:
            return "Low confidence - may need verification"
    
    def handle_user_feedback(
        self,
        image_id: str,
        cv_prediction: str,
        confidence: float,
        user_confirmed: bool,
        correct_landmark: Optional[str] = None
    ) -> Dict:
        """
        Handle user confirmation feedback.
        
        Args:
            image_id: Unique identifier for the image
            cv_prediction: What CV model predicted
            confidence: CV model confidence
            user_confirmed: True if user confirmed CV prediction, False if rejected
            correct_landmark: If rejected, what the correct landmark is (from LLaVA or user)
            
        Returns:
            Routing decision for next step:
            {
                'action': 'accept_cv' | 'use_llava',
                'cv_was_correct': bool,
                'should_save_feedback': bool
            }
        """
        feedback_entry = {
            'image_id': image_id,
            'cv_prediction': cv_prediction,
            'confidence': confidence,
            'user_confirmed': user_confirmed,
            'correct_landmark': correct_landmark,
            'timestamp': None  # Add timestamp in production
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Save to file if path provided
        if self.feedback_log_path and user_confirmed is False:
            self._save_feedback(feedback_entry)
        
        if user_confirmed:
            # User confirmed CV prediction - accept it
            return {
                'action': 'accept_cv',
                'cv_was_correct': True,
                'should_save_feedback': False
            }
        else:
            # User rejected - need LLaVA fallback
            return {
                'action': 'use_llava',
                'cv_was_correct': False,
                'should_save_feedback': True
            }
    
    def _save_feedback(self, feedback_entry: Dict):
        """Save feedback to file for later analysis."""
        try:
            feedback_path = Path(self.feedback_log_path)
            
            # Load existing feedback if file exists
            if feedback_path.exists():
                with open(feedback_path, 'r') as f:
                    existing = json.load(f)
            else:
                existing = []
            
            existing.append(feedback_entry)
            
            with open(feedback_path, 'w') as f:
                json.dump(existing, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save feedback: {e}")
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics from user feedback history."""
        if not self.feedback_history:
            return {
                'total_predictions': 0,
                'user_confirmed': 0,
                'user_rejected': 0,
                'cv_accuracy': 0.0
            }
        
        total = len(self.feedback_history)
        confirmed = sum(1 for f in self.feedback_history if f['user_confirmed'])
        rejected = total - confirmed
        
        # Calculate accuracy by confidence bucket
        high_conf = [f for f in self.feedback_history if f['confidence'] >= 0.80]
        med_conf = [f for f in self.feedback_history if 0.60 <= f['confidence'] < 0.80]
        low_conf = [f for f in self.feedback_history if f['confidence'] < 0.60]
        
        return {
            'total_predictions': total,
            'user_confirmed': confirmed,
            'user_rejected': rejected,
            'cv_accuracy': (confirmed / total * 100) if total > 0 else 0,
            'high_conf_accuracy': (sum(1 for f in high_conf if f['user_confirmed']) / len(high_conf) * 100) if high_conf else 0,
            'med_conf_accuracy': (sum(1 for f in med_conf if f['user_confirmed']) / len(med_conf) * 100) if med_conf else 0,
            'low_conf_accuracy': (sum(1 for f in low_conf if f['user_confirmed']) / len(low_conf) * 100) if low_conf else 0
        }
    
    def predict_batch(
        self,
        images: List[Image.Image],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Predict for batch of images with confidence routing.
        
        Args:
            images: List of PIL Images
            top_k: Number of top predictions per image
            
        Returns:
            List of routing decisions (one per image)
        """
        return [self.predict(img, top_k) for img in images]
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate routing statistics from batch predictions.
        
        Args:
            results: List of routing decisions from predict() or predict_batch()
            
        Returns:
            Statistics dictionary with routing breakdown
        """
        total = len(results)
        
        return_cv = sum(1 for r in results if r['action'] == 'return_cv')
        verify = sum(1 for r in results if r['action'] == 'verify_with_llava')
        use_llava = sum(1 for r in results if r['action'] == 'use_llava')
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        return {
            'total_predictions': total,
            'instant_cv': return_cv,
            'verify_with_llava': verify,
            'use_llava_only': use_llava,
            'instant_cv_pct': (return_cv / total * 100) if total > 0 else 0,
            'verify_pct': (verify / total * 100) if total > 0 else 0,
            'llava_only_pct': (use_llava / total * 100) if total > 0 else 0,
            'avg_confidence': avg_confidence,
            'needs_llava_total': verify + use_llava,
            'needs_llava_pct': ((verify + use_llava) / total * 100) if total > 0 else 0
        }


def test_confidence_router():
    """Test the confidence router on validation set."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    # Paths
    model_path = "data/checkpoints/landmark_detector_500classes_best.pth"
    class_mapping_path = "data/landmarks_500class/class_mapping.json"
    val_manifest_path = "data/landmarks_500class/val_manifest.json"
    
    # Load validation data
    with open(val_manifest_path, 'r') as f:
        val_data = json.load(f)
    
    # Initialize router
    print("\n" + "="*60)
    print("CONFIDENCE ROUTER TEST")
    print("="*60 + "\n")
    
    router = ConfidenceRouter(
        model_path=model_path,
        class_mapping_path=class_mapping_path,
        high_confidence_threshold=0.80,
        low_confidence_threshold=0.60
    )
    
    # Test on subset of validation set
    print(f"\nTesting on {min(1000, len(val_data))} validation images...\n")
    
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = []
    correct_by_action = {'return_cv': 0, 'verify_with_llava': 0, 'use_llava': 0}
    total_by_action = {'return_cv': 0, 'verify_with_llava': 0, 'use_llava': 0}
    
    for i, item in enumerate(val_data[:1000]):
        if i % 100 == 0:
            print(f"  Processed {i}/1000 images...")
        
        try:
            img_path = Path(item['image_path'])
            if not img_path.is_absolute():
                img_path = Path("data/landmarks_500class") / img_path
            
            image = Image.open(img_path).convert('RGB')
            
            # Get routing decision
            result = router.predict(image, top_k=5)
            results.append(result)
            
            # Check if prediction is correct
            true_landmark = router.idx_to_landmark[item['class_idx']]
            is_correct = (result['top_prediction'] == true_landmark)
            
            action = result['action']
            total_by_action[action] += 1
            if is_correct:
                correct_by_action[action] += 1
                
        except Exception as e:
            print(f"  Warning: Skipped image {i}: {e}")
            continue
    
    # Calculate statistics
    stats = router.get_statistics(results)
    
    # Print results
    print("\n" + "="*60)
    print("ROUTING STATISTICS")
    print("="*60 + "\n")
    
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Average confidence: {stats['avg_confidence']:.1%}\n")
    
    print(f"Instant CV (>{router.high_threshold*100:.0f}% conf):")
    print(f"  Count: {stats['instant_cv']} ({stats['instant_cv_pct']:.1f}%)")
    if total_by_action['return_cv'] > 0:
        acc = correct_by_action['return_cv'] / total_by_action['return_cv'] * 100
        print(f"  Accuracy: {acc:.1f}%")
    
    print(f"\nVerify with LLaVA ({router.low_threshold*100:.0f}-{router.high_threshold*100:.0f}% conf):")
    print(f"  Count: {stats['verify_with_llava']} ({stats['verify_pct']:.1f}%)")
    if total_by_action['verify_with_llava'] > 0:
        acc = correct_by_action['verify_with_llava'] / total_by_action['verify_with_llava'] * 100
        print(f"  Accuracy: {acc:.1f}%")
    
    print(f"\nUse LLaVA only (<{router.low_threshold*100:.0f}% conf):")
    print(f"  Count: {stats['use_llava_only']} ({stats['llava_only_pct']:.1f}%)")
    if total_by_action['use_llava'] > 0:
        acc = correct_by_action['use_llava'] / total_by_action['use_llava'] * 100
        print(f"  CV accuracy (would be wrong): {acc:.1f}%")
    
    print(f"\nTotal needing LLaVA: {stats['needs_llava_total']} ({stats['needs_llava_pct']:.1f}%)")
    
    print("\n" + "="*60)
    print("CONFIDENCE DISTRIBUTION")
    print("="*60 + "\n")
    
    confidences = [r['confidence'] for r in results]
    print(f"Min: {min(confidences):.1%}")
    print(f"25th percentile: {np.percentile(confidences, 25):.1%}")
    print(f"Median: {np.median(confidences):.1%}")
    print(f"75th percentile: {np.percentile(confidences, 75):.1%}")
    print(f"Max: {max(confidences):.1%}")
    
    print("\n✓ Confidence router test complete!\n")


if __name__ == '__main__':
    test_confidence_router()
