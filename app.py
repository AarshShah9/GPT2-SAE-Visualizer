import streamlit as st
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
import plotly.express as px
import pandas as pd

def get_device():
    """
    Automatically detect the best available device for inference.
    In the order: CUDA (NVIDIA GPUs) => MPS (Apple Silicon) => CPU.
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@st.cache_resource
def load_models():
    """
    Load GPT-2 Small and its corresponding Sparse Autoencoder.
    
    This function is cached using @st.cache_resource to ensure models
    are only loaded once per session, significantly improving performance.
    
    Returns:
        tuple: (model, sae, device) where:
            - model: HookedTransformer instance of GPT-2 Small
            - sae: Sparse Autoencoder trained on Layer 8 residual stream
            - device: str indicating which device models are loaded on
    
    Raises:
        Exception: If model download or loading fails
    """
    try:
        # Detect the best available device
        device = get_device()
        
        # Load GPT-2 Small using TransformerLens
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            device=device
        )
        
        # Load pre-trained SAE for Layer 8 residual stream
        sae = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
            device=device
        )
        
        return model, sae, device
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please check your internet connection and try again.")
        raise e


def analyze_text(text, model, sae):
    """
    Run the input text through GPT-2 and extract SAE feature activations with token attribution.
    
    Args:
        text (str): Input text to analyze
        model: HookedTransformer model
        sae: Sparse Autoencoder model
    
    Returns:
        list: List of dictionaries containing feature information:
            - feature_id: int, the feature index
            - max_activation: float, the maximum activation value
            - triggering_token: str, the token that caused the highest activation
            - token_position: int, the position of the triggering token
            - all_tokens: list of str, all tokens in the input
            - token_activations: list of float, activation for this feature at each token position
    """
    # Run the text through GPT-2 with caching enabled
    # This captures intermediate activations at all layers
    with torch.no_grad():
        _logits, cache = model.run_with_cache(text)
        
        # Get the tokens for interpretability
        tokens = model.to_str_tokens(text)
        
        # Extract Layer 8 residual stream activations
        # Shape: (batch_size, seq_len, d_model)
        activations = cache["blocks.8.hook_resid_pre"]
        
        # Pass activations through the SAE encoder
        # This decomposes the dense activations into sparse features
        # Shape: (batch_size, seq_len, n_features)
        feature_acts = sae.encode(activations)
        
        # IMPORTANT: Exclude the BOS token (position 0) which is <|endoftext|>
        # GPT-2 prepends this token and it dominates feature activations
        # We want features that activate on actual content, not the special BOS token
        
        # Only consider content tokens (skip position 0)
        content_feature_acts = feature_acts[:, 1:, :]  # Shape: (batch, seq_len-1, n_features)
        content_tokens = tokens[1:]  # Skip the BOS token
        
        # Find the maximum activation for each feature across content tokens only
        # Shape: (batch_size, n_features)
        max_acts_per_feature = content_feature_acts.max(dim=1).values
        
        # Get the top 5 most active features (based on content tokens)
        top_values, top_indices = max_acts_per_feature.topk(5, dim=1)
        
        # For each top feature, find which content token activated it most
        feature_info = []
        for feat_idx, max_val in zip(top_indices[0], top_values[0]):
            # Get activations for this feature across content tokens only
            token_acts = content_feature_acts[0, :, feat_idx]  # Shape: (seq_len-1,)
            
            # Find which content token had the highest activation
            max_token_idx = token_acts.argmax()
            max_token = content_tokens[max_token_idx]
            max_activation = token_acts[max_token_idx].item()
            
            feature_info.append({
                'feature_id': feat_idx.item(),
                'max_activation': max_activation,
                'triggering_token': max_token,
                'token_position': max_token_idx.item() + 1,  # +1 to account for skipped BOS
                'all_tokens': content_tokens,  # Only show content tokens
                'token_activations': token_acts.cpu().tolist()
            })
        
        return feature_info


def main():
    """Main Streamlit application."""
    
    # Set page configuration
    st.set_page_config(
        page_title="GPT-2 SAE Visualizer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("GPT-2 Sparse Autoencoder Visualizer")
    st.markdown("""
    This app demonstrates **Mechanistic Interpretability** by showing how a Sparse Autoencoder (SAE) 
    decomposes GPT-2's internal activations into interpretable features.
    
    **How it works:**
    1. Input text is processed by GPT-2 Small
    2. We extract the residual stream activations at Layer 8
    3. The SAE decomposes these dense activations into sparse, interpretable features
    4. We visualize the most active features
    """)
    
    # Sidebar for model status
    with st.sidebar:
        st.header("System Status")
        
        # Load models (cached)
        try:
            model, sae, device = load_models()
            st.success("Models Loaded")
            st.info(f"**Device:** {device.upper()}")
            st.info(f"**Model:** GPT-2 Small")
            st.info(f"**SAE:** Layer 8 Residual")
            
        except Exception as e:
            st.error("Failed to load models")
            st.stop()
        
        st.markdown("---")
        st.markdown("""
        ### About SAEs
        Sparse Autoencoders learn to represent neural network 
        activations as sparse combinations of interpretable features.
        This helps us understand what information flows through the network.
        """)
    
    # Main content area
    st.header("Input Text")
    
    # Text input area
    input_text = st.text_area(
        "Enter text to analyze:",
        value="The quick brown fox jumps over the lazy dog",
        height=100,
        help="Enter any text you'd like to analyze. The SAE will identify which features activate most strongly."
    )
    
    # Analyze button
    analyze_button = st.button("Analyze Features", type="primary", width="stretch")
    
    if analyze_button:
        if not input_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            # Show progress
            with st.spinner("Analyzing text and extracting features..."):
                try:
                    # Run analysis
                    feature_info = analyze_text(input_text, model, sae)
                    
                    # Display results
                    st.success("Analysis complete!")
                    
                    # Feature Dashboard
                    st.header("Top 5 Active Features")
                    st.markdown("""
                    These are the SAE features that activated most strongly for your input text.
                    Each feature corresponds to a learned pattern in the model's internal representations.
                    """)
                    
                    # Create columns for metrics with token attribution
                    cols = st.columns(5)
                    for i, feat in enumerate(feature_info):
                        with cols[i]:
                            st.metric(
                                label=f"Feature #{feat['feature_id']}",
                                value=f"{feat['max_activation']:.3f}",
                                help=f"Activation strength: {feat['max_activation']:.6f}"
                            )
                            # Show triggering token
                            st.caption(f"Triggered by: **{feat['triggering_token']}**")
                            st.caption(f"Position: {feat['token_position']}")
                    
                    # Create bar chart
                    st.subheader("Feature Activation Strengths")
                    
                    # Prepare data for plotting
                    chart_data = pd.DataFrame({
                        'Feature': [f"Feature #{feat['feature_id']}" for feat in feature_info],
                        'Activation': [feat['max_activation'] for feat in feature_info],
                        'Token': [feat['triggering_token'] for feat in feature_info]
                    })
                    
                    # Create interactive Plotly bar chart
                    fig = px.bar(
                        chart_data,
                        x='Feature',
                        y='Activation',
                        title='Top 5 SAE Feature Activations',
                        labels={'Activation': 'Activation Strength', 'Feature': 'SAE Feature ID'},
                        color='Activation',
                        color_continuous_scale='Viridis',
                        hover_data={'Token': True}
                    )
                    
                    fig.update_layout(
                        xaxis_title="Feature ID",
                        yaxis_title="Activation Strength",
                        showlegend=False,
                        height=500
                    )
                    
                    st.plotly_chart(fig, key="feature_chart")
                    
                    # Token-level activation details
                    st.subheader("Token-Level Feature Activation")
                    st.markdown("""
                    Below you can see exactly which tokens (words/subwords) in your input text 
                    triggered each of the top features. This helps understand what patterns the SAE learned.
                    """)
                    
                    # Create expandable sections for each feature
                    for feat in feature_info:
                        with st.expander(f"Feature #{feat['feature_id']} - Triggered by '{feat['triggering_token']}'"):
                            # Show all tokens with their activation values
                            st.write("**Activation per token:**")
                            
                            # Create a mini dataframe for this feature
                            token_df = pd.DataFrame({
                                'Position': range(len(feat['all_tokens'])),
                                'Token': feat['all_tokens'],
                                'Activation': feat['token_activations']
                            })
                            
                            # Display as a table
                            st.dataframe(
                                token_df.style.highlight_max(subset=['Activation'], color='lightgreen'),
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            # Mini line chart for this feature
                            token_chart = px.line(
                                token_df,
                                x='Position',
                                y='Activation',
                                title=f'Feature #{feat["feature_id"]} Activation Across Tokens',
                                markers=True,
                                hover_data={'Token': True}
                            )
                            token_chart.update_layout(height=300)
                            st.plotly_chart(token_chart, key=f"token_chart_{feat['feature_id']}")
                    
                    # Additional info
                    st.info("""
                    **Interpretation Note:** These feature indices correspond to learned patterns 
                    in GPT-2's internal representations. The tokens shown above reveal which specific 
                    words or concepts activated each feature.
                    
                    In a full interpretability suite, each feature would have human-readable descriptions 
                    (e.g., "Feature #1024: Names of animals").
                    """)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.error("Please try again with different text or check the logs.")


if __name__ == "__main__":
    main()

