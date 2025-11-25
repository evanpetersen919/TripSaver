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
from core.recommendation_engine import RecommendationEngine


@st.cache_resource(ttl=None, show_spinner="Loading vision models...")
def load_pipeline():
    """Load the vision pipeline with all models (cached)"""
    base_path = Path(__file__).parent
    model_path = str(base_path / "data" / "checkpoints" / "landmark_detector_100classes_best.pth")
    landmark_names_path = str(base_path / "data" / "checkpoints" / "landmark_names_100classes.json")
    
    # Force reload by clearing any cached models
    pipeline = VisionPipeline(
        enable_llava=True,
        enable_clip=True,  # Now works without FAISS index
        enable_landmark=True,
        landmark_weights_path=model_path,
        landmark_names_path=landmark_names_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    )
    
    # Verify models loaded correctly
    if pipeline.llava_analyzer:
        print(f"✓ LLaVA analyzer verified")
    
    return pipeline


@st.cache_resource
def load_recommendation_engine():
    """Load recommendation engine (cached)"""
    return RecommendationEngine()


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
    
    # Add button to clear cache and reload models on GPU
    if st.sidebar.button("🔄 Reload Models on GPU", help="Clear cache and reinitialize all models on GPU"):
        st.cache_resource.clear()
        st.rerun()
    
    st.title("🎯 Computer Vision Pipeline Demo")
    st.markdown("""
    Upload an image to analyze it with **multiple AI models in parallel**:
    - 🏛️ **Landmark Detector** - Identifies 100 landmarks (91.45% accuracy)
    - 🤖 **LLaVA AI** - Advanced vision-language analysis and location reasoning
    - 🔍 **CLIP Embedder** - Visual similarity search for finding similar-looking places
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
        
        st.markdown("### 🤖 LLaVA Vision AI")
        st.markdown("""
        **Model:** LLaVA-1.5-7B  
        **Type:** Vision-Language Model  
        **Capability:** Natural language scene understanding  
        **Features:** Location analysis, descriptions, Q&A
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
                models_used = sum([1 for k in ['llava_analyzer', 'landmark_detector', 'clip_embedder'] if results.get(k)])
                st.metric("Models Used", f"{models_used}/3")
            
            st.markdown("---")
            
            # Get smart recommendation strategy
            strategy = pipeline.get_recommendation_strategy(results)
            
            # Show user-friendly result based on strategy
            if strategy['mode'] == 'landmark':
                st.success(f"### ✅ {strategy['user_message']}")
                st.metric("Confidence", f"{strategy['confidence']:.1%}")
                
                # Add rejection button for high confidence predictions
                if st.button("❌ This is incorrect - Show alternatives", key="reject_landmark"):
                    st.warning("### 🤔 Let me show you other possibilities:")
                    alternatives = strategy.get('alternatives', [])
                    if alternatives:
                        for i, alt in enumerate(alternatives, 2):  # Start from 2 since 1 was the rejected one
                            st.markdown(f"{i}. **{alt}**")
                    st.info("💡 Tip: This helps us improve! Consider providing feedback.")
                
            elif strategy['mode'] == 'landmark_options':
                st.warning(f"### 🤔 {strategy['user_message']}")
                st.markdown("**Top 3 possibilities:**")
                for i, (option, conf) in enumerate(zip(strategy['options'], strategy['confidences']), 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"{i}. **{option}**")
                    with col2:
                        st.caption(f"{conf:.1%}")
                        
            elif strategy['mode'] == 'scene':
                st.info(f"### 🔍 {strategy['user_message']}")
                
                # Search database using LLaVA keywords
                if strategy.get('search_keywords'):
                    rec_engine = load_recommendation_engine()
                    search_results = rec_engine.search_by_description(
                        llava_description=strategy['search_keywords'],
                        clip_embedding=strategy.get('clip_embedding'),
                        top_k=5,
                        min_similarity=0.3
                    )
                    
                    if search_results:
                        # Check how many have visual embeddings
                        with_visual = sum(1 for r in search_results if r['visual_similarity'] > 0)
                        st.markdown(f"**🎯 Matching landmarks found:** (🎨 {with_visual}/{len(search_results)} with visual similarity)")
                        
                        for i, result in enumerate(search_results, 1):
                            with st.expander(f"{i}. {result['name']} — Score: {result['final_score']:.0%}", expanded=i<=3):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"📍 **{result['country']}**")
                                    if result.get('description'):
                                        st.caption(result['description'][:200] + "...")
                                with col2:
                                    st.metric("Combined Score", f"{result['final_score']:.0%}")
                                    st.caption(f"📝 Text: {result['text_similarity']:.0%}")
                                    if result['visual_similarity'] > 0:
                                        st.caption(f"🎨 Visual: {result['visual_similarity']:.0%}")
                                    else:
                                        st.caption("🎨 Visual: N/A (no embedding)")
                    else:
                        st.warning("No matching landmarks found in database.")
                
            else:  # exploration mode
                st.info(f"### 🌍 {strategy['user_message']}")
            
            st.markdown("---")
            
            # Tabs for detailed results
            tab1, tab2, tab3 = st.tabs(["🏛️ Landmark Details", "🤖 LLaVA Analysis", "📊 All Predictions"])
            
            with tab1:
                if results.get('landmark_detector'):
                    landmark = results['landmark_detector']
                    
                    # Show confidence level
                    level = landmark.get('confidence_level', 'unknown')
                    level_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(level, '⚪')
                    st.markdown(f"**Confidence Level:** {level_emoji} {level.upper()}")
                    st.caption(f"Took {landmark['elapsed_ms']:.1f}ms")
                    
                    st.markdown("#### All Predictions")
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
                if results.get('llava_analyzer'):
                    llava = results['llava_analyzer']
                    st.success("**AI Analysis Complete**")
                    st.metric("Inference Time", f"{llava['elapsed_ms']:.0f}ms")
                    
                    st.markdown("#### 📝 Location Analysis")
                    st.info(llava['description'])
                    
                    st.markdown("#### 🔍 Analysis Type")
                    st.caption(f"Type: {llava['type']}")
                    
                    # Show word cloud or key phrases
                    if llava['description']:
                        st.markdown("#### 🏷️ Key Information")
                        # Simple keyword extraction
                        words = llava['description'].split()
                        if len(words) > 20:
                            st.caption(f"Analysis contains {len(words)} words of detailed description")
                else:
                    st.warning("LLaVA analyzer not available")
            
            with tab3:
                # CLIP Embedder Results
                if results.get('clip_embedder'):
                    clip = results['clip_embedder']
                    st.success("**CLIP Visual Embedding Generated**")
                    st.metric("Inference Time", f"{clip['elapsed_ms']:.0f}ms")
                    
                    st.markdown("#### 🎨 Visual Embedding")
                    st.info(f"Generated {clip['embedding_dim']}-dimensional visual embedding for similarity search")
                    
                    st.markdown("#### 💡 What is CLIP?")
                    st.caption("""
                    CLIP (Contrastive Language-Image Pre-Training) encodes images into a vector space 
                    where visually similar images are close together. This enables finding landmarks that 
                    *look* similar, complementing text-based search.
                    """)
                    
                    # Show embedding stats
                    if clip.get('embedding') is not None:
                        import numpy as np
                        emb = clip['embedding']
                        st.markdown("#### 📊 Embedding Statistics")
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Mean", f"{np.mean(emb):.4f}")
                        with cols[1]:
                            st.metric("Std Dev", f"{np.std(emb):.4f}")
                        with cols[2]:
                            st.metric("L2 Norm", f"{np.linalg.norm(emb):.4f}")
                else:
                    st.info("CLIP visual embeddings are being used in recommendations")
                
                st.markdown("---")
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
            
            # Itinerary Recommendations Section
            if results.get('llava_analyzer'):
                st.markdown("---")
                st.markdown("### 🗺️ Find Similar Places")
                st.markdown("Get personalized recommendations based on this image's features")
                
                # Itinerary input
                rec_engine = load_recommendation_engine()
                available_landmarks = rec_engine.get_available_landmarks()
                
                # Global search option
                search_globally = st.checkbox(
                    "🌍 Search globally (no itinerary needed)",
                    value=False,
                    help="Find similar places anywhere in the world"
                )
                
                if not search_globally:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        selected_landmarks = st.multiselect(
                            "Select landmarks in your itinerary:",
                            options=available_landmarks,
                            default=[],
                            help="Choose landmarks you're planning to visit"
                        )
                    
                    with col2:
                        max_distance = st.number_input(
                            "Max distance (km):",
                            min_value=5,
                            max_value=200,
                            value=50,
                            step=5
                        )
                else:
                    selected_landmarks = []
                    max_distance = None
                
                # Enable button if global search OR landmarks selected
                can_search = search_globally or len(selected_landmarks) > 0
                
                if st.button("🔍 Get Recommendations", type="primary", disabled=not can_search):
                    spinner_text = "Finding similar places worldwide..." if search_globally else "Finding similar places near your itinerary..."
                    with st.spinner(spinner_text):
                        try:
                            llava_desc = results['llava_analyzer']['description']
                            
                            # Get CLIP embedding if available
                            clip_embedding = None
                            if 'clip_embedder' in results and results['clip_embedder']:
                                clip_embedding = results['clip_embedder'].get('embedding')
                            
                            recommendations = rec_engine.recommend(
                                itinerary_landmarks=selected_landmarks if not search_globally else [],
                                llava_description=llava_desc,
                                max_distance_km=max_distance if not search_globally else None,
                                clip_embedding=clip_embedding,
                                top_k=10 if search_globally else 5
                            )
                            
                            if recommendations:
                                st.success(f"Found {len(recommendations)} recommendations!")
                                
                                for i, rec in enumerate(recommendations, 1):
                                    with st.expander(f"#{i} {rec.name} — Score: {rec.final_score:.2f}", expanded=i<=3):
                                        cols = st.columns([2, 1])
                                        
                                        with cols[0]:
                                            st.markdown(f"**📍 Location:** {rec.country}")
                                            if not search_globally:
                                                st.markdown(f"**📏 Distance:** {rec.distance_km:.1f}km from {rec.closest_itinerary_item}")
                                            st.markdown(f"**🎯 Similarity:** {rec.similarity_score:.0%} match")
                                            if rec.description:
                                                st.caption(f"💬 {rec.description}")
                                        
                                        with cols[1]:
                                            st.metric("Final Score", f"{rec.final_score:.2f}")
                                            st.metric("Coordinates", f"{rec.latitude:.4f}, {rec.longitude:.4f}")
                                
                            else:
                                if search_globally:
                                    st.warning("No similar landmarks found")
                                else:
                                    st.warning(f"No landmarks found within {max_distance}km of your itinerary")
                        
                        except Exception as e:
                            st.error(f"Error generating recommendations: {str(e)}")
                
                elif not selected_landmarks:
                    st.info("💡 Select landmarks from your itinerary to get personalized recommendations")
        
        else:
            st.info("👆 Upload an image to get started")
            
            st.markdown("### What This Pipeline Does")
            st.markdown("""
            This demo runs **3 computer vision models simultaneously**:
            
            **🏛️ Landmark Detector**
            - Identifies 100 landmarks from Google Landmarks Dataset
            - Examples: Eiffel Tower, Golden Gate Bridge, Grand Canyon
            - Best for: specific architectural landmarks and monuments
            
            **🤖 LLaVA Vision AI**  
            - Advanced vision-language model with location reasoning
            - Provides natural language descriptions and analysis
            - Can identify landmarks and suggest likely locations
            - Best for: contextual understanding and unknown locations
            
            **🔍 CLIP Embedder**
            - Generates visual embeddings for similarity search
            - Finds landmarks that *look* similar (architecture, scenery, vibe)
            - Best for: visual matching beyond text descriptions
            
            The pipeline runs all models **in parallel** and combines their predictions
            for comprehensive location analysis and recommendations.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit • Powered by EfficientNet-B3 + LLaVA-1.5-7B + OpenCLIP ViT-B/32</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
