"""
Streamlit Demo for Vision Pipeline

Interactive web interface to test all CV models in parallel.
Upload images and see predictions from Scene Classifier, CLIP Embedder, and Landmark Detector.

Author: Evan Petersen
Date: November 2025
"""

import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import torch
import plotly.graph_objects as go
import plotly.express as px
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.vision_pipeline import VisionPipeline


@st.cache_resource
def load_pipeline():
    """Load the vision pipeline with all models (cached)"""
    model_path = "data/checkpoints/landmark_detector_100classes_best.pth"
    landmark_names_path = "data/checkpoints/landmark_names_100classes.json"
    
    # Force reload by clearing any cached models
    pipeline = VisionPipeline(
        enable_scene=True,
        enable_clip=False,  # Enable when FAISS index is available
        enable_landmark=True,
        landmark_weights_path=model_path,
        landmark_names_path=landmark_names_path
    )
    
    # Verify scene classifier loaded correctly
    if pipeline.scene_classifier:
        assert len(pipeline.scene_classifier.categories) == 365, "Scene categories mismatch!"
        print(f"✓ Scene classifier verified: {len(pipeline.scene_classifier.categories)} categories")
    
    return pipeline


def create_confidence_chart(predictions, title="Top-5 Predictions"):
    """Create a horizontal bar chart of predictions"""
    labels = [pred.get('landmark') or pred.get('category', 'Unknown') for pred in predictions]
    confidences = [pred['confidence'] * 100 for pred in predictions]
    
    fig = go.Figure(go.Bar(
        x=confidences,
        y=labels,
        orientation='h',
        marker=dict(
            color=confidences,
            colorscale='Viridis',
            showscale=False
        ),
        text=[f'{c:.1f}%' for c in confidences],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Confidence (%)",
        yaxis_title="",
        height=300,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="Vision Pipeline Demo",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Computer Vision Pipeline Demo")
    st.markdown("""
    Upload an image to analyze it with **multiple AI models in parallel**:
    - 🏛️ **Landmark Detector** - Identifies 100 landmarks (91.45% accuracy)
    - 🌄 **Scene Classifier** - Recognizes 365 scene types (beach, mountain, city, etc.)
    - 🔍 **CLIP Embedder** - Visual similarity search (coming soon)
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Models Active")
        
        st.markdown("### 🏛️ Landmark Detector")
        st.markdown("""
        **Architecture:** EfficientNet-B3  
        **Classes:** 100 landmarks  
        **Top-1 Accuracy:** 91.45%  
        **Top-5 Accuracy:** 96.99%  
        **Training:** Optuna-optimized
        """)
        
        st.markdown("### 🌄 Scene Classifier")
        st.markdown("""
        **Architecture:** ResNet-18  
        **Classes:** 365 scenes  
        **Dataset:** Places365  
        **Examples:** beach, forest, kitchen, office
        """)
        
        st.markdown("---")
        st.markdown("### ⚡ Performance")
        st.markdown("""
        All models run **in parallel** for maximum speed.
        Typical inference: ~100-300ms total
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of a famous landmark"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Image info
            st.caption(f"Size: {image.size[0]} x {image.size[1]} pixels")
    
    with col2:
        if uploaded_file is not None:
            st.header("Analysis Results")
            
            # Load pipeline
            with st.spinner('Loading AI models...'):
                pipeline = load_pipeline()
            
            # Run prediction
            with st.spinner('Running inference...'):
                start_time = time.time()
                results = pipeline.predict(image)
                inference_time = (time.time() - start_time) * 1000
            
            # Performance metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Inference Time", f"{inference_time:.0f}ms")
            with col_b:
                models_used = sum([1 for k in ['scene_classifier', 'landmark_detector', 'clip_embedder'] if results.get(k)])
                st.metric("Models Used", f"{models_used}/3")
            
            st.markdown("---")
            
            # Tabs for different models
            tab1, tab2, tab3 = st.tabs(["🏛️ Landmark", "🌄 Scene", "📊 Summary"])
            
            with tab1:
                if results.get('landmark_detector'):
                    landmark = results['landmark_detector']
                    st.success(f"**Detected:** {landmark['top_landmark']}")
                    st.metric("Confidence", f"{landmark['confidence']:.2%}", 
                             help=f"Model took {landmark['elapsed_ms']:.1f}ms")
                    
                    st.markdown("#### Top-5 Predictions")
                    for i, pred in enumerate(landmark['predictions'], 1):
                        cols = st.columns([0.5, 3, 2])
                        with cols[0]:
                            st.markdown(f"**#{i}**")
                        with cols[1]:
                            st.markdown(f"`{pred['landmark']}`")
                        with cols[2]:
                            st.progress(pred['confidence'])
                            st.caption(f"{pred['confidence']:.2%}")
                    
                    fig = create_confidence_chart(landmark['predictions'], "Landmark Predictions")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Landmark detector not available")
            
            with tab2:
                if results.get('scene_classifier'):
                    scene = results['scene_classifier']
                    st.success(f"**Scene Type:** {scene['top_scene']}")
                    st.metric("Confidence", f"{scene['confidence']:.2%}",
                             help=f"Model took {scene['elapsed_ms']:.1f}ms")
                    
                    st.markdown("#### Top-5 Scene Categories")
                    for i, pred in enumerate(scene['predictions'], 1):
                        cols = st.columns([0.5, 3, 2])
                        with cols[0]:
                            st.markdown(f"**#{i}**")
                        with cols[1]:
                            st.markdown(f"`{pred['category']}`")
                        with cols[2]:
                            st.progress(pred['confidence'])
                            st.caption(f"{pred['confidence']:.2%}")
                    
                    fig = create_confidence_chart(scene['predictions'], "Scene Classifications")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Scene classifier not available")
            
            with tab3:
                st.markdown("### Aggregated Prediction")
                aggregated = pipeline.aggregate_predictions(results)
                
                st.info(f"**Location Type:** {aggregated['location_type']}")
                st.info(f"**Location Name:** {aggregated['location_name']}")
                st.metric("Overall Confidence", f"{aggregated['confidence']:.2%}")
                
                st.markdown("#### Evidence from Models")
                for evidence in aggregated['evidence']:
                    with st.expander(f"📍 {evidence['source'].replace('_', ' ').title()}"):
                        st.markdown(f"**Prediction:** {evidence['value']}")
                        st.markdown(f"**Confidence:** {evidence['confidence']:.2%}")
                
                # Timing breakdown
                st.markdown("#### Performance Breakdown")
                timing_data = []
                if results.get('landmark_detector'):
                    timing_data.append({'Model': 'Landmark Detector', 'Time (ms)': results['landmark_detector']['elapsed_ms']})
                if results.get('scene_classifier'):
                    timing_data.append({'Model': 'Scene Classifier', 'Time (ms)': results['scene_classifier']['elapsed_ms']})
                if results.get('clip_embedder'):
                    timing_data.append({'Model': 'CLIP Embedder', 'Time (ms)': results['clip_embedder']['elapsed_ms']})
                
                if timing_data:
                    import pandas as pd
                    df = pd.DataFrame(timing_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
        
        else:
            st.info("👆 Upload an image to get started")
            
            st.markdown("### What This Pipeline Does")
            st.markdown("""
            This demo runs **3 computer vision models simultaneously**:
            
            **🏛️ Landmark Detector**
            - Identifies 100 landmarks from Google Landmarks Dataset
            - Examples: Sant Julià de Pedra, The Beresford, Gulf State Park
            - Best for: architectural landmarks, monuments, parks
            
            **🌄 Scene Classifier**  
            - Recognizes 365 different scene types
            - Examples: beach, forest, kitchen, stadium, church
            - Best for: general location context
            
            **🔍 CLIP Embedder** (Coming Soon)
            - Finds visually similar images in a database
            - Best for: niche locations not in training data
            
            The pipeline runs all models **in parallel** and combines their predictions
            for the most accurate location detection possible.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit • Powered by EfficientNet-B3 + ResNet-18 + CLIP</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
