"""
Gradio app with user confirmation flow for landmark detection.

User Flow:
1. Upload image
2. See CV prediction instantly (200ms) with top-5 options
3. Click "✓ Correct" or "✗ Wrong"
4. If correct: Done (save result)
5. If wrong: Run LLaVA fallback (8-15s) for accurate result
"""

import gradio as gr
from PIL import Image
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.confidence_router import ConfidenceRouter


class LandmarkDetectionApp:
    """Gradio app with confirmation UI."""
    
    def __init__(self):
        # Initialize confidence router
        model_path = "data/checkpoints/landmark_detector_500classes_best.pth"
        class_mapping_path = "data/landmarks_500class/class_mapping.json"
        feedback_log_path = "data/user_feedback.json"
        
        self.router = ConfidenceRouter(
            model_path=model_path,
            class_mapping_path=class_mapping_path,
            feedback_log_path=feedback_log_path
        )
        
        self.current_result = None
        self.current_image_id = 0
    
    def predict_initial(self, image: Image.Image) -> tuple:
        """
        Step 1: Show CV prediction with confirmation UI.
        
        Returns: (prediction_text, top_5_text, confidence_badge, show_confirm_buttons)
        """
        if image is None:
            return "Please upload an image", "", "", gr.update(visible=False)
        
        # Get CV prediction
        result = self.router.predict(image, top_k=5, mode='confirmation')
        
        self.current_result = result
        self.current_image_id += 1
        
        # Format prediction text
        prediction = result['top_prediction']
        confidence = result['confidence']
        recommendation = result['recommendation']
        
        # Main prediction display
        prediction_text = f"## 📍 {prediction}\n\n**Confidence:** {confidence:.1%} - *{recommendation}*"
        
        # Top-5 alternatives
        top_5_list = "\n".join([
            f"{i+1}. **{name}** ({conf:.1%})"
            for i, (name, conf) in enumerate(result['top_k_predictions'])
        ])
        top_5_text = f"### Other possibilities:\n{top_5_list}"
        
        # Confidence badge color
        if confidence >= 0.80:
            badge = "🟢 High confidence"
        elif confidence >= 0.60:
            badge = "🟡 Medium confidence"
        else:
            badge = "🔴 Low confidence"
        
        return (
            prediction_text,
            top_5_text,
            badge,
            gr.update(visible=True)  # Show confirm buttons
        )
    
    def handle_confirmation(self, confirmed: bool, image: Image.Image) -> tuple:
        """
        Step 2: Handle user confirmation.
        
        Args:
            confirmed: True if user clicked "✓ Correct", False if "✗ Wrong"
            image: Current image (for LLaVA if needed)
            
        Returns: (final_result_text, feedback_stats)
        """
        if self.current_result is None:
            return "No prediction to confirm", ""
        
        # Log feedback
        feedback = self.router.handle_user_feedback(
            image_id=str(self.current_image_id),
            cv_prediction=self.current_result['top_prediction'],
            confidence=self.current_result['confidence'],
            user_confirmed=confirmed,
            correct_landmark=None if confirmed else "pending_llava"
        )
        
        if confirmed:
            # User confirmed - we're done!
            result_text = f"""
            ### ✅ Confirmed!
            
            **Result:** {self.current_result['top_prediction']}
            
            Response time: ~200ms
            """
        else:
            # User rejected - need LLaVA fallback
            result_text = f"""
            ### 🔄 Running advanced analysis...
            
            The CV model suggested: {self.current_result['top_prediction']}
            
            *[In production, LLaVA would run here for 8-15 seconds]*
            
            **For now:** Marked for manual review
            """
            # TODO: Integrate actual LLaVA call here
        
        # Get feedback stats
        stats = self.router.get_feedback_stats()
        stats_text = f"""
        ### 📊 Session Stats
        - Total predictions: {stats['total_predictions']}
        - CV accuracy: {stats['cv_accuracy']:.1f}%
        - Confirmed: {stats['user_confirmed']}
        - Rejected: {stats['user_rejected']}
        """
        
        return result_text, stats_text
    
    def build_interface(self):
        """Build Gradio interface."""
        
        with gr.Blocks(title="Landmark Detection with Confirmation") as app:
            gr.Markdown("# 🗺️ Landmark Detection")
            gr.Markdown("Upload an image and confirm if the prediction is correct!")
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Image")
                    predict_btn = gr.Button("🔍 Detect Landmark", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    # Initial prediction display
                    prediction_display = gr.Markdown("*Prediction will appear here*")
                    confidence_badge = gr.Markdown("")
                    top_5_display = gr.Markdown("")
                    
                    # Confirmation buttons (hidden initially)
                    with gr.Row(visible=False) as confirm_buttons:
                        confirm_btn = gr.Button("✓ Correct", variant="primary", size="lg")
                        reject_btn = gr.Button("✗ Wrong", variant="stop", size="lg")
                    
                    # Final result display
                    gr.Markdown("---")
                    final_result = gr.Markdown("")
                    feedback_stats = gr.Markdown("")
            
            # Event handlers
            predict_btn.click(
                fn=self.predict_initial,
                inputs=[image_input],
                outputs=[prediction_display, top_5_display, confidence_badge, confirm_buttons]
            )
            
            confirm_btn.click(
                fn=lambda img: self.handle_confirmation(True, img),
                inputs=[image_input],
                outputs=[final_result, feedback_stats]
            )
            
            reject_btn.click(
                fn=lambda img: self.handle_confirmation(False, img),
                inputs=[image_input],
                outputs=[final_result, feedback_stats]
            )
            
            # Examples
            gr.Markdown("---")
            gr.Markdown("### 💡 How it works:")
            gr.Markdown("""
            1. **Upload** your landmark image
            2. **Review** the instant CV prediction (~200ms)
            3. **Confirm** if correct, or **reject** if wrong
            4. If wrong: Advanced LLaVA model runs for accurate detection (~12s)
            
            **Benefits:**
            - Fast for common landmarks (you confirm)
            - Accurate for tricky cases (LLaVA fallback)
            - Learns from your feedback over time
            """)
        
        return app


def main():
    """Launch the Gradio app."""
    app = LandmarkDetectionApp()
    interface = app.build_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == '__main__':
    main()
