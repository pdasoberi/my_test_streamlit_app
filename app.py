import os
import streamlit as st
from openai import OpenAI
import httpx

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"], http_client=httpx.Client(),)

# Verify API key is available
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please set it as an environment variable.")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    st.write("Adjust the behavior of the AI model:")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    st.write("Lower values make responses more focused, higher values make them more creative.")
    if st.button("🗑️ Reset Conversation"):
        st.session_state.conversation = []

# Initialize session state for conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Helper functions for the agent's multi-step process
def rephrase_prompt(user_question):
    return f"Rephrase the following question in your own words using RAR (Role, Action, Result) and best practices in prompt engineering: {user_question}"

def answer_prompt(rephrased_question):
    return f"Answer the following question concisely: {rephrased_question}"

def reasoning_prompt(original_question, rephrased_question, answer):
    return (
        f"Let's think through whether this answer is correct step by step:\n\n"
        f"Original Question: {original_question}\n"
        f"Rephrased Question: {rephrased_question}\n"
        f"Answer: {answer}\n\n"
        f"Please provide a detailed step-by-step reasoning about whether this answer is correct and complete."
    )

def evaluation_prompt(reasoning):
    return (
        f"Based on this reasoning:\n{reasoning}\n\n"
        f"Make a final judgment: Is the answer correct and complete?\n"
        f"Respond only with 'yes' or 'no'."
    )

def call_openai(prompt, temperature=0.7, stream=True):
    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=stream
        )

        if stream:
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    chunk_text = chunk.choices[0].delta.content
                    full_response += chunk_text
                    yield full_response
        else:
            return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {e}"

# Main chat interface
user_input = st.chat_input("Ask me anything!")

if user_input:
    # Add user input to the conversation
    st.session_state.conversation.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Step 1: Rephrase the user question
    with st.chat_message("assistant"):
        st.write("Rephrasing your question...")
        message_placeholder = st.empty()
        rephrased_question = ""
        for response in call_openai(rephrase_prompt(user_input), temperature=temperature):
            rephrased_question = response
            message_placeholder.markdown(rephrased_question + "▌")
        message_placeholder.markdown(rephrased_question)

    st.session_state.conversation.append({"role": "assistant", "content": f"Rephrased Question: {rephrased_question}"})

    # Step 2: Answer the rephrased question
    with st.chat_message("assistant"):
        st.write("Answering your question...")
        message_placeholder = st.empty()
        answer = ""
        for response in call_openai(answer_prompt(rephrased_question), temperature=temperature):
            answer = response
            message_placeholder.markdown(answer + "▌")
        message_placeholder.markdown(answer)

    st.session_state.conversation.append({"role": "assistant", "content": f"Answer: {answer}"})

    # Step 3: Verify the answer
    with st.chat_message("assistant"):
        st.write("Reasoning about the answer...")
        message_placeholder = st.empty()
        reasoning = ""
        for response in call_openai(reasoning_prompt(user_input, rephrased_question, answer), temperature=temperature):
            reasoning = response
            message_placeholder.markdown(reasoning + "▌")
        message_placeholder.markdown(reasoning)

        st.write("Making final evaluation...")
        verification = ""
        for response in call_openai(evaluation_prompt(reasoning), temperature=0.5, stream=True):
            verification = response
        st.write(f"Verification Result: {verification}")

    st.session_state.conversation.append({"role": "assistant", "content": f"Reasoning: {reasoning}"})
    st.session_state.conversation.append({"role": "assistant", "content": f"Verification: {verification}"})

    # Step 4: Repeat if verification fails
    while verification.strip().lower() == "no":
        st.session_state.conversation.append({"role": "assistant", "content": "Rephrasing and retrying..."})

        # Rephrase again
        with st.chat_message("assistant"):
            st.write("Rephrasing your question again...")
            message_placeholder = st.empty()
            rephrased_question = ""
            for response in call_openai(rephrase_prompt(user_input), temperature=temperature):
                rephrased_question = response
                message_placeholder.markdown(rephrased_question + "▌")
            message_placeholder.markdown(rephrased_question)

        st.session_state.conversation.append({"role": "assistant", "content": f"Rephrased Question: {rephrased_question}"})

        # Answer again
        with st.chat_message("assistant"):
            st.write("Answering your question again...")
            message_placeholder = st.empty()
            answer = ""
            for response in call_openai(answer_prompt(rephrased_question), temperature=temperature):
                answer = response
                message_placeholder.markdown(answer + "▌")
            message_placeholder.markdown(answer)

        st.session_state.conversation.append({"role": "assistant", "content": f"Answer: {answer}"})

        # Verify again
        with st.chat_message("assistant"):
            st.write("Reasoning about the answer again...")
            message_placeholder = st.empty()
            reasoning = ""
            for response in call_openai(reasoning_prompt(user_input, rephrased_question, answer), temperature=temperature):
                reasoning = response
                message_placeholder.markdown(reasoning + "▌")
            message_placeholder.markdown(reasoning)

            st.write("Making final evaluation...")
            verification = ""
            for response in call_openai(evaluation_prompt(reasoning), temperature=0.5, stream=True):
                verification = response
            st.write(f"Verification Result: {verification}")

        st.session_state.conversation.append({"role": "assistant", "content": f"Reasoning: {reasoning}"})
        st.session_state.conversation.append({"role": "assistant", "content": f"Verification: {verification}"})

    # Display the final answer
    with st.chat_message("assistant"):
        st.write(f"Final Answer: {answer}")
    st.session_state.conversation.append({"role": "assistant", "content": f"Final Answer: {answer}"})
