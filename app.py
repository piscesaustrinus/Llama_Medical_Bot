import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Initialize Streamlit app title
st.title("Simple Chatbot")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

def display_message(role, content):
    with st.chat_message(role):
        st.markdown(content)

# Display existing chat history
for role, content in st.session_state.history:
    display_message(role, content)

# Load model and tokenizer just once
@st.cache_resource
def load_model():
    model_name = "distilgpt2"  # Small, free model for testing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load the model pipeline
text_generator = load_model()

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Generate AI response using the pipeline
    response = text_generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
    ai_response = response[len(prompt):].strip()  # Remove input prompt from generated text
    
    if ai_response:
        # Display user message
        display_message("user", prompt)

        # Display AI response
        display_message("assistant", ai_response)

        # Append messages to history
        st.session_state.history.append(("user", prompt))
        st.session_state.history.append(("assistant", ai_response))
