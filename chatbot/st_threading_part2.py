import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    return uuid.uuid4()

def generate_topic_from_conversation(messages):
    """
    Generate a short conversation topic using the entire conversation so far.
    messages: list of {'role': 'user'/'assistant', 'content': '...'}
    """
    # Convert message history into a readable dialogue for the LLM
    conv_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
    topic_prompt = (
        "You are a helpful AI that generates short, meaningful conversation titles.\n"
        "Given the conversation below, create a title in **max 5 words** that summarizes the main topic.\n\n"
        f"{conv_text}\n\n"
        "Title:"
    )

    response = chatbot.invoke(
        {"messages": [HumanMessage(content=topic_prompt)]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}}  # temporary thread for topic generation
    )

    # Handle dict output safely
    if isinstance(response, dict):
        if "messages" in response:  # Case 1
            last_msg = response["messages"][-1]
            return last_msg.content.strip() if hasattr(last_msg, "content") else str(last_msg).strip()

        if "values" in response and "messages" in response["values"]:  # Case 2
            last_msg = response["values"]["messages"][-1]
            return last_msg.content.strip() if hasattr(last_msg, "content") else str(last_msg).strip()

    return "New Chat"

def reset_chat():
    """Resets chat and creates a new thread."""
    thread_id = generate_thread_id()
    add_thread(thread_id, topic="New Chat")
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []

def add_thread(thread_id, topic="New Chat"):
    """Adds a thread with a topic if not already in the session."""
    if not any(thread['id'] == thread_id for thread in st.session_state['chat_threads']):
        st.session_state['chat_threads'].append({'id': thread_id, 'topic': topic})

def load_conversation(thread_id):
    """Loads a conversation's messages from LangGraph backend."""
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])  # add initial thread

# **************************************** Sidebar UI *********************************
st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat', key="new_chat_btn"):
    reset_chat()

st.sidebar.header('My Conversations')

for thread in reversed(st.session_state['chat_threads']):
    if st.sidebar.button(thread['topic'], key=f"btn_{thread['id']}"):
        st.session_state['thread_id'] = thread['id']
        messages = load_conversation(thread['id'])

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages

# **************************************** Main UI ************************************
# Show conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # Append user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Generate topic if this is the first or second user message (gives more context)
    user_msgs_count = len([m for m in st.session_state['message_history'] if m['role'] == 'user'])
    if user_msgs_count in (1, 2):
        topic = generate_topic_from_conversation(st.session_state['message_history'])
        for thread in st.session_state['chat_threads']:
            if thread['id'] == st.session_state['thread_id']:
                thread['topic'] = topic
                break

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Stream assistant response
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    # Append assistant response
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
