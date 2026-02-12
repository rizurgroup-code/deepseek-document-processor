import streamlit as st
import requests
import pdfplumber
import docx
import chardet
from io import BytesIO
import time

# ============================================
# PAGE CONFIGURATION - MUST BE FIRST COMMAND
# ============================================
st.set_page_config(
    page_title="DeepSeek Document Processor",
    page_icon="📄",
    layout="wide"
)

# ============================================
# CUSTOM CSS FOR PROFESSIONAL LOOK
# ============================================
def inject_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 1rem;
        }
        .uploader-box {
            border: 2px dashed #3B82F6;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            background-color: #F9FAFB;
        }
        .message-bubble {
            border-radius: 1rem;
            padding: 1.5rem;
            margin: 1rem 0;
            background-color: #F3F4F6;
            border-left: 4px solid #3B82F6;
        }
        .success-badge {
            color: #10B981;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],              # Chat history
        "api_key": "",              # DeepSeek API key
        "uploaded_files_content": [], # Processed file content
        "current_mode": "deepseek-chat", # Model selection
        "temperature": 0.3,         # Response creativity
        "system_prompt": "You are an expert technical document analyst and code generator. Process the provided documents and complete the user's task exactly as instructed."
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================
# FILE PROCESSING FUNCTIONS
# ============================================
def extract_text_from_pdf(file_bytes):
    """Extract text from PDF files"""
    text = ""
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"PDF processing error: {e}")
    return text

def extract_text_from_docx(file_bytes):
    """Extract text from Word documents"""
    text = ""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"DOCX processing error: {e}")
    return text

def extract_text_from_txt(file_bytes):
    """Extract text from text files with encoding detection"""
    try:
        encoding = chardet.detect(file_bytes)['encoding']
        text = file_bytes.decode(encoding or 'utf-8', errors='ignore')
    except:
        text = file_bytes.decode('utf-8', errors='ignore')
    return text

def process_uploaded_file(uploaded_file):
    """Route file to appropriate extractor based on type"""
    file_bytes = uploaded_file.read()
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    
    if "pdf" in file_type:
        text = extract_text_from_pdf(file_bytes)
    elif "word" in file_type or "docx" in file_type:
        text = extract_text_from_docx(file_bytes)
    elif "text" in file_type or "txt" in file_type:
        text = extract_text_from_txt(file_bytes)
    else:
        text = f"[Unsupported file type: {file_type}]"
    
    return {
        "name": file_name,
        "content": text[:50000],  # Limit to 50k chars to stay within token limits
        "size": len(text)
    }

# ============================================
# DEEPSEEK API CALL FUNCTION
# ============================================
def call_deepseek_api(user_message, file_contents, api_key):
    """Send request to DeepSeek API with document context"""
    
    # Prepare the context from uploaded files
    context = ""
    if file_contents:
        context = "=== DOCUMENTS ===\n\n"
        for idx, file in enumerate(file_contents, 1):
            context += f"--- Document {idx}: {file['name']} ---\n"
            context += file['content'][:8000]  # Send first 8000 chars per file
            context += "\n\n"
    
    # Construct the full prompt
    if context:
        full_prompt = f"""{st.session_state.system_prompt}

{context}

=== USER TASK ===
{user_message}

Process the above documents and complete the task. Generate the requested output (code, HTML, technical document, etc.) directly without commentary unless asked for explanation."""
    else:
        full_prompt = user_message
    
    # API call configuration
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": st.session_state.current_mode,
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "temperature": st.session_state.temperature,
        "stream": False
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

# ============================================
# UI RENDERING FUNCTIONS
# ============================================
def render_sidebar():
    """Render the sidebar with all controls"""
    with st.sidebar:
        st.markdown("## ⚙️ Control Center")
        st.markdown("---")
        
        # ===== API KEY SECTION =====
        st.markdown("### 🔑 API Configuration")
        
        # Try to get from secrets first, then allow manual input
        default_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        api_key_input = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=st.session_state.api_key or default_key,
            placeholder="sk-...",
            help="Your API key from platform.deepseek.com (funded via WeChat Pay)"
        )
        
        if api_key_input:
            st.session_state.api_key = api_key_input
        
        # Show validation hint
        if st.session_state.api_key:
            if st.session_state.api_key.startswith("sk-"):
                st.success("✅ API key format valid")
            else:
                st.warning("⚠️ API key should start with 'sk-'")
        else:
            st.error("❌ API key required")
        
        st.markdown("---")
        
        # ===== MODEL SETTINGS =====
        st.markdown("### 🧠 Model Settings")
        
        st.session_state.current_mode = st.selectbox(
            "Model",
            options=["deepseek-chat", "deepseek-reasoner"],
            index=0,
            help="deepseek-chat: general tasks, deepseek-reasoner: complex reasoning"
        )
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Lower = precise, Higher = creative"
        )
        
        st.markdown("---")
        
        # ===== SYSTEM PROMPT =====
        st.markdown("### 📝 System Prompt")
        
        st.session_state.system_prompt = st.text_area(
            "Instructions to AI",
            value=st.session_state.system_prompt,
            height=100,
            help="This defines how the AI behaves"
        )
        
        st.markdown("---")
        
        # ===== FILE UPLOAD =====
        st.markdown("### 📁 Document Upload")
        st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process newly uploaded files
        if uploaded_files:
            with st.spinner("Processing documents..."):
                new_files_content = []
                for file in uploaded_files:
                    processed = process_uploaded_file(file)
                    new_files_content.append(processed)
                    st.caption(f"✅ {processed['name']} - {processed['size']} chars")
                
                # Update session state
                st.session_state.uploaded_files_content = new_files_content
        
        # Show currently loaded files
        if st.session_state.uploaded_files_content:
            st.markdown("#### Loaded Documents:")
            for file in st.session_state.uploaded_files_content:
                st.caption(f"📄 {file['name']} ({file['size']} chars)")
            
            if st.button("🗑️ Clear All Documents"):
                st.session_state.uploaded_files_content = []
                st.rerun()
        
        st.markdown("---")
        
        # ===== NEW CHAT BUTTON =====
        if st.button("✨ New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def render_main_chat():
    """Render the main chat interface"""
    
    # Header
    st.markdown('<h1 class="main-header">📄 DeepSeek Document Processor</h1>', unsafe_allow_html=True)
    st.markdown("*Upload documents, describe your task, get results.*")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Describe what you want to do with your documents..."):
        # Check prerequisites
        if not st.session_state.api_key:
            st.error("⚠️ Please enter your DeepSeek API key in the sidebar")
            st.stop()
        
        if not st.session_state.api_key.startswith("sk-"):
            st.error("⚠️ Invalid API key format. Should start with 'sk-'")
            st.stop()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Processing with DeepSeek API..."):
                response = call_deepseek_api(
                    prompt,
                    st.session_state.uploaded_files_content,
                    st.session_state.api_key
                )
                st.markdown(response)
                
                # Add to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================
# MAIN APP EXECUTION
# ============================================
def main():
    render_sidebar()
    render_main_chat()

if __name__ == "__main__":
    main()