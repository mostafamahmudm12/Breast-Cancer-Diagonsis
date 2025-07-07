import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ML Model Prediction Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .status-success {
        color: #28a745;
        font-weight: 600;
    }
    .status-error {
        color: #dc3545;
        font-weight: 600;
    }
    .prediction-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_status' not in st.session_state:
    st.session_state.api_status = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def check_api_status(base_url: str, api_key: str) -> Dict[str, Any]:
    """Check if the API is running and accessible"""
    try:
        headers = {'X-API-Key': api_key}
        response = requests.get(f"{base_url}/", headers=headers, timeout=10)
        
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        elif response.status_code == 403:
            return {"status": "error", "message": "Invalid API key"}
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to API server"}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Request timeout"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

def make_prediction(base_url: str, api_key: str, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a prediction request to the API"""
    try:
        headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f"{base_url}/predict/{model_name}",
            headers=headers,
            json=input_data,
            timeout=30
        )
        
        if response.status_code == 200:
            return {"status": "success", "data": response.json()}
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Prediction request timeout"}
    except Exception as e:
        return {"status": "error", "message": f"Prediction error: {str(e)}"}

def main():
    # Main header
    st.markdown('<h1 class="main-header">🤖 ML Model Prediction Interface</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        st.subheader("API Settings")
        base_url = st.text_input(
            "API Base URL",
            value="http://localhost:8000",
            help="The base URL of your FastAPI server"
        )
        
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Your API key for authentication"
        )
        
        # Test API connection
        if st.button("🔍 Test API Connection"):
            if not api_key:
                st.error("Please enter your API key")
            else:
                with st.spinner("Testing API connection..."):
                    result = check_api_status(base_url, api_key)
                    st.session_state.api_status = result
        
        # Display API status
        if st.session_state.api_status:
            if st.session_state.api_status["status"] == "success":
                st.success("✅ API Connected Successfully")
                data = st.session_state.api_status["data"]
                st.info(f"**App:** {data.get('app_name', 'Unknown')}\n\n**Version:** {data.get('version', 'Unknown')}\n\n**Status:** {data.get('status', 'Unknown')}")
            else:
                st.error(f"❌ API Connection Failed: {st.session_state.api_status['message']}")
        
        st.divider()
        
        # Model Configuration
        st.subheader("Model Settings")
        model_name = st.text_input(
            "Model Name",
            value="default_model",
            help="Name of the model to use for predictions"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">📊 Input Data</h2>', unsafe_allow_html=True)
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "JSON Input", "CSV Upload"],
            horizontal=True
        )
        
        input_data = None
        
        if input_method == "Manual Input":
            st.subheader("Manual Data Entry")
            
            # Dynamic form for manual input
            with st.form("manual_input_form"):
                st.write("Enter your data (you can add multiple rows):")
                
                # Initialize data container
                if 'manual_data' not in st.session_state:
                    st.session_state.manual_data = [{}]
                
                # Number of samples
                num_samples = st.number_input("Number of samples", min_value=1, max_value=100, value=1)
                
                # Adjust the data list based on number of samples
                while len(st.session_state.manual_data) < num_samples:
                    st.session_state.manual_data.append({})
                while len(st.session_state.manual_data) > num_samples:
                    st.session_state.manual_data.pop()
                
                # Create input fields for each sample
                samples = []
                for i in range(num_samples):
                    st.write(f"**Sample {i+1}:**")
                    sample_data = {}
                    
                    # You can customize these fields based on your model's expected input
                    col_a, col_b = st.columns(2)
                    with col_a:
                        feature1 = st.number_input(f"Feature 1 (Sample {i+1})", key=f"f1_{i}")
                        feature2 = st.number_input(f"Feature 2 (Sample {i+1})", key=f"f2_{i}")
                    with col_b:
                        feature3 = st.number_input(f"Feature 3 (Sample {i+1})", key=f"f3_{i}")
                        feature4 = st.number_input(f"Feature 4 (Sample {i+1})", key=f"f4_{i}")
                    
                    sample_data = {
                        "feature1": feature1,
                        "feature2": feature2,
                        "feature3": feature3,
                        "feature4": feature4
                    }
                    samples.append(sample_data)
                
                if st.form_submit_button("Prepare Data"):
                    input_data = {"samples": samples}
        
        elif input_method == "JSON Input":
            st.subheader("JSON Data Input")
            
            # Example JSON structure
            example_json = {
                "samples": [
                    {
                        "feature1": 1.0,
                        "feature2": 2.0,
                        "feature3": 3.0,
                        "feature4": 4.0
                    },
                    {
                        "feature1": 5.0,
                        "feature2": 6.0,
                        "feature3": 7.0,
                        "feature4": 8.0
                    }
                ]
            }
            
            st.code(json.dumps(example_json, indent=2), language="json")
            
            json_input = st.text_area(
                "Enter JSON data:",
                height=200,
                placeholder="Paste your JSON data here..."
            )
            
            if st.button("Parse JSON"):
                if json_input:
                    try:
                        input_data = json.loads(json_input)
                        st.success("✅ JSON parsed successfully!")
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Invalid JSON: {str(e)}")
                else:
                    st.warning("Please enter JSON data")
        
        elif input_method == "CSV Upload":
            st.subheader("CSV Data Upload")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("**Preview of uploaded data:**")
                    st.dataframe(df.head())
                    
                    if st.button("Convert to Input Format"):
                        # Convert DataFrame to the expected input format
                        samples = df.to_dict('records')
                        input_data = {"samples": samples}
                        st.success(f"✅ Converted {len(samples)} samples from CSV")
                        
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {str(e)}")
        
        # Prediction section
        if input_data:
            st.markdown('<h2 class="section-header">🚀 Make Prediction</h2>', unsafe_allow_html=True)
            
            # Display input data preview
            with st.expander("📋 Input Data Preview"):
                st.json(input_data)
            
            if st.button("🔮 Make Prediction", type="primary"):
                if not api_key:
                    st.error("Please enter your API key in the sidebar")
                elif not model_name:
                    st.error("Please enter a model name in the sidebar")
                else:
                    with st.spinner("Making prediction..."):
                        result = make_prediction(base_url, api_key, model_name, input_data)
                        
                        if result["status"] == "success":
                            st.success("✅ Prediction completed successfully!")
                            
                            # Store in history
                            st.session_state.prediction_history.append({
                                "timestamp": pd.Timestamp.now(),
                                "model": model_name,
                                "input": input_data,
                                "output": result["data"]
                            })
                            
                            # Display results
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.subheader("🎯 Prediction Results")
                            st.json(result["data"])
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                        else:
                            st.error(f"❌ Prediction failed: {result['message']}")
    
    with col2:
        st.markdown('<h2 class="section-header">📈 Results & History</h2>', unsafe_allow_html=True)
        
        # Prediction History
        if st.session_state.prediction_history:
            st.subheader("Recent Predictions")
            
            # Show last 5 predictions
            recent_predictions = st.session_state.prediction_history[-5:]
            
            for i, pred in enumerate(reversed(recent_predictions)):
                with st.expander(f"Prediction {len(recent_predictions) - i} - {pred['timestamp'].strftime('%H:%M:%S')}"):
                    st.write(f"**Model:** {pred['model']}")
                    st.write(f"**Time:** {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write("**Input:**")
                    st.json(pred['input'])
                    st.write("**Output:**")
                    st.json(pred['output'])
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.success("History cleared!")
        else:
            st.info("No predictions made yet. Make your first prediction to see results here!")
        
        # Statistics
        if st.session_state.prediction_history:
            st.subheader("📊 Statistics")
            total_predictions = len(st.session_state.prediction_history)
            st.metric("Total Predictions", total_predictions)
            
            # Model usage stats
            model_counts = {}
            for pred in st.session_state.prediction_history:
                model = pred['model']
                model_counts[model] = model_counts.get(model, 0) + 1
            
            st.write("**Model Usage:**")
            for model, count in model_counts.items():
                st.write(f"- {model}: {count} predictions")

if __name__ == "__main__":
    main()