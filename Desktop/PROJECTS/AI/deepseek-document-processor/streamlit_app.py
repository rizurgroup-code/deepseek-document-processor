import streamlit as st
import requests
import pdfplumber
import docx
import chardet
from io import BytesIO
import json
import time

# ============================================
# PAGE CONFIGURATION - MUST BE FIRST COMMAND
# ============================================
st.set_page_config(
    page_title="DeepSeek Document Intelligence",
    page_icon="🧠",
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
        .stExpander {
            border-left: 3px solid #F59E0B;
            background-color: #FFFBEB;
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
        "messages": [],                # Chat history
        "api_key": "",                # DeepSeek API key
        "uploaded_files_content": [], # Processed file content
        "current_mode": "deepseek-chat",  # Model selection
        "temperature": 0.3,           # Response creativity
        "system_prompt": "You are an expert technical document analyst and code generator. Process the provided documents and complete the user's task exactly as instructed. When generating code, include comments and error handling. When summarizing, be concise and highlight key points.",
        "documents_attached": False,  # Track if docs already sent in conversation
        "stream": True,              # Enable streaming by default
        "template": "Custom (no template)"  # Selected prompt template
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
    
    # Limit text length to avoid token overflow (50k chars ~= 15k tokens)
    truncated_text = text[:50000]
    
    return {
        "name": file_name,
        "content": truncated_text,
        "size": len(text)
    }

# ============================================
# DEEPSEEK API CALL FUNCTIONS
# ============================================
def call_deepseek_api_stream(payload, headers):
    """Streaming version – yields chunks with reasoning/content"""
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=120
    )
    
    reasoning_collected = ""
    content_collected = ""
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    
                    # Reasoning content (specific to deepseek-reasoner)
                    if "reasoning_content" in delta:
                        reasoning_chunk = delta["reasoning_content"]
                        reasoning_collected += reasoning_chunk
                        yield {"type": "reasoning", "content": reasoning_chunk}
                    
                    # Normal content
                    if "content" in delta:
                        content_chunk = delta["content"]
                        content_collected += content_chunk
                        yield {"type": "content", "content": content_chunk}
                        
                except json.JSONDecodeError:
                    continue
                except KeyError:
                    continue
    
    yield {"type": "done", "reasoning": reasoning_collected, "content": content_collected}

def call_deepseek_api(messages, api_key, model="deepseek-chat", temperature=0.3, stream=False):
    """Send conversation to DeepSeek API with support for reasoning traces"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream
    }
    
    try:
        if stream:
            return call_deepseek_api_stream(payload, headers)
        else:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                choice = result["choices"][0]
                
                # Extract reasoning if present (DeepSeek-R1)
                reasoning = choice.get("message", {}).get("reasoning_content", "")
                content = choice["message"]["content"]
                
                return {
                    "content": content,
                    "reasoning": reasoning
                }
            else:
                return {
                    "content": f"❌ API Error: {response.status_code} - {response.text}",
                    "reasoning": ""
                }
                
    except Exception as e:
        return {
            "content": f"❌ Connection Error: {str(e)}",
            "reasoning": ""
        }

# ============================================
# CONVERSATION BUILDER (MEMORY & DOCUMENTS)
# ============================================
def build_conversation_messages(new_user_prompt):
    """Construct the full message list for the API call with proper memory"""
    
    messages = []
    
    # 1. System prompt (defines behavior)
    system_msg = {
        "role": "system", 
        "content": st.session_state.system_prompt
    }
    messages.append(system_msg)
    
    # 2. Document context - send only ONCE at the beginning of conversation
    if st.session_state.uploaded_files_content and not st.session_state.documents_attached:
        context = "=== DOCUMENTS ===\n\n"
        for idx, file in enumerate(st.session_state.uploaded_files_content, 1):
            context += f"--- Document {idx}: {file['name']} ---\n"
            context += file['content'][:8000] + "\n\n"  # Send first 8000 chars per file
        
        messages.append({
            "role": "user",
            "content": f"Please process the following documents. They are attached for reference throughout our conversation:\n\n{context}"
        })
        # Assistant acknowledgment – creates natural flow
        messages.append({
            "role": "assistant",
            "content": "I have loaded the documents. I will refer to them as needed. Please tell me what you'd like me to do."
        })
        st.session_state.documents_attached = True
    
    # 3. Conversation history (exclude reasoning entries to keep API clean)
    for msg in st.session_state.messages:
        if msg["role"] in ["user", "assistant"]:
            messages.append(msg)
    
    # 4. Current user prompt (if not already the last message)
    if not messages or messages[-1]["role"] != "user" or messages[-1]["content"] != new_user_prompt:
        messages.append({"role": "user", "content": new_user_prompt})
    
    return messages

# ============================================
# UI SIDEBAR
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
        
        if st.session_state.api_key:
            if st.session_state.api_key.startswith("sk-"):
                st.success("✅ API key valid")
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
            help="deepseek-chat: fast, general tasks. deepseek-reasoner: shows reasoning, better for complex problems."
        )
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Lower = more precise, higher = more creative"
        )
        
        st.session_state.stream = st.checkbox(
            "Stream responses",
            value=st.session_state.stream,
            help="Show words as they are generated (faster feel)"
        )
        
        st.markdown("---")
        
        # ===== SYSTEM PROMPT =====
        st.markdown("### 📝 System Prompt")
        
        st.session_state.system_prompt = st.text_area(
            "Instructions to AI",
            value=st.session_state.system_prompt,
            height=120,
            help="This defines how the AI behaves"
        )
        
        st.markdown("---")
        
        # ===== PROMPT TEMPLATES =====
        st.markdown("### 📋 Task Templates")
        template_options = [
            "Custom (no template)",
            "Generate Python code",
            "Generate HTML webpage",
            "Summarize document",
            "Commercial proposal",
            "Extract tables to Markdown",
            "Refactor code",
            "Create technical documentation",
            "Analyze contract clauses"
        ]
        
        selected_template = st.selectbox(
            "Quick start templates",
            template_options,
            index=template_options.index(st.session_state.template) if st.session_state.template in template_options else 0
        )
        
        # Apply template if changed
        if selected_template != st.session_state.template:
            st.session_state.template = selected_template
            if selected_template == "Generate Python code":
                st.session_state.system_prompt = "You are an expert Python developer. Write clean, efficient, well-documented code based on the requirements. Include error handling, comments, and type hints where appropriate. Output only the code block unless explanation is specifically requested."
                st.session_state.temperature = 0.2
            elif selected_template == "Generate HTML webpage":
                st.session_state.system_prompt = "You are a front-end developer. Create responsive, modern HTML5/CSS3 webpages. Include appropriate meta tags, semantic structure, mobile-friendly design, and sample content. Provide complete HTML file ready to run."
                st.session_state.temperature = 0.3
            elif selected_template == "Summarize document":
                st.session_state.system_prompt = "You are a technical writer. Summarize documents clearly and concisely. Capture key points, decisions, and action items. Use bullet points and headings for readability. Keep the summary to about 10-20% of the original length."
                st.session_state.temperature = 0.4
            elif selected_template == "Commercial proposal":
                st.session_state.system_prompt = "You are a business consultant. Create professional commercial proposals based on the provided specifications. Include executive summary, scope, deliverables, timeline, pricing structure, and terms. Format with clear sections."
                st.session_state.temperature = 0.5
            elif selected_template == "Extract tables to Markdown":
                st.session_state.system_prompt = "You extract tabular data from documents and convert it to clean Markdown table format. Preserve all rows and columns accurately. If no tables are present, state that clearly."
                st.session_state.temperature = 0.1
            elif selected_template == "Refactor code":
                st.session_state.system_prompt = "You are a senior software engineer. Refactor the provided code to improve readability, performance, and maintainability. Follow best practices and design patterns. Explain the changes you made."
                st.session_state.temperature = 0.2
            elif selected_template == "Create technical documentation":
                st.session_state.system_prompt = "You are a technical writer. Create comprehensive documentation from code or specifications. Include overview, installation, usage examples, API reference, and troubleshooting. Use Markdown formatting."
                st.session_state.temperature = 0.3
            elif selected_template == "Analyze contract clauses":
                st.session_state.system_prompt = "You are a legal document analyst. Review the contract and identify key clauses, obligations, risks, and missing elements. Highlight unusual terms and provide plain-language explanations."
                st.session_state.temperature = 0.2
        
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
                # Reset document attachment flag so new docs are sent in next message
                st.session_state.documents_attached = False
        
        # Show currently loaded documents
        if st.session_state.uploaded_files_content:
            st.markdown("#### Loaded Documents:")
            for file in st.session_state.uploaded_files_content:
                st.caption(f"📄 {file['name']} ({file['size']} chars)")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Documents", use_container_width=True):
                    st.session_state.uploaded_files_content = []
                    st.session_state.documents_attached = False
                    st.rerun()
            with col2:
                if st.button("🔄 Re-attach", use_container_width=True):
                    st.session_state.documents_attached = False
                    st.success("Documents will be re-sent in next message!")
        
        st.markdown("---")
        
        # ===== NEW CHAT BUTTON =====
        if st.button("✨ New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.documents_attached = False  # Docs need to be re-sent
            st.rerun()

# ============================================
# MAIN CHAT INTERFACE
# ============================================
def render_main_chat():
    """Render the main chat interface"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 DeepSeek Document Intelligence</h1>', unsafe_allow_html=True)
    st.markdown("*Upload documents, describe your task, get results with full reasoning.*")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
        elif message["role"] == "assistant_reasoning":
            # Show reasoning in expander (for non-streaming history)
            with st.chat_message("assistant"):
                with st.expander("🧠 Deep Thinking (previous)", expanded=False):
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
            with st.spinner("Thinking..." if st.session_state.current_mode == "deepseek-reasoner" else "Processing..."):
                
                # Build messages list with conversation memory
                messages = build_conversation_messages(prompt)
                
                # Call API with appropriate streaming mode
                response_data = call_deepseek_api(
                    messages=messages,
                    api_key=st.session_state.api_key,
                    model=st.session_state.current_mode,
                    temperature=st.session_state.temperature,
                    stream=st.session_state.stream
                )
                
                # Handle streaming vs non-streaming
                if st.session_state.stream:
                    reasoning_placeholder = st.empty()
                    content_placeholder = st.empty()
                    
                    reasoning_text = ""
                    content_text = ""
                    final_reasoning = ""
                    final_content = ""
                    
                    for chunk in response_data:
                        if chunk["type"] == "reasoning":
                            reasoning_text += chunk["content"]
                            with reasoning_placeholder.expander("🧠 Deep Thinking", expanded=False):
                                st.markdown(reasoning_text + "▌")
                        elif chunk["type"] == "content":
                            content_text += chunk["content"]
                            content_placeholder.markdown(content_text + "▌")
                        elif chunk["type"] == "done":
                            final_reasoning = chunk["reasoning"]
                            final_content = chunk["content"]
                            if reasoning_text:
                                with reasoning_placeholder.expander("🧠 Deep Thinking", expanded=False):
                                    st.markdown(final_reasoning)
                            content_placeholder.markdown(final_content)
                    
                else:
                    # Non-streaming
                    result = response_data
                    final_content = result["content"]
                    final_reasoning = result["reasoning"]
                    
                    if final_reasoning:
                        with st.expander("🧠 Deep Thinking", expanded=False):
                            st.markdown(final_reasoning)
                    st.markdown(final_content)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": final_content})
                if final_reasoning:
                    st.session_state.messages.append({"role": "assistant_reasoning", "content": final_reasoning})

# ============================================
# MAIN APP EXECUTION
# ============================================
def main():
    render_sidebar()
    render_main_chat()

if __name__ == "__main__":
    main()