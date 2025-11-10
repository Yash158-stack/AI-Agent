import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate

# Build the chat template once globally
prompt_template = ChatPromptTemplate.from_template(
    """
    You are an AI assistant who answers based strictly on the given context.
    If the answer is not found in the context, reply: "I couldn't find relevant info in the uploaded documents."
are 
    You should:
        - Maintain natural, human-like flow in conversation.
        - Use the chat history to recall details (like the user's name or previous topics).
        - Be concise — don't restate obvious things or overexplain.
        - Answer only from the provided document context when relevant.
        - If a question is unrelated to the documents, politely say so — but still respond helpfully.
        - Avoid repeating your name or previous messages unless necessary.
        - Keep your tone warm, confident, and approachable.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}
    """
)

def handle_conversation(user_query, retriever, chat_history):
    """Handles one turn of conversation using Gemini and document retrieval."""

    # ✅ No need to call genai.configure() here again (already done in app.py)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Retrieve relevant docs
    context_docs = retriever.invoke(user_query)
    context = "\n\n".join(d.page_content for d in context_docs)

    # Prepare chat history string
    chat_hist_text = "\n".join(f"{r}: {m}" for r, m in chat_history)

    # Build final prompt
    prompt = prompt_template.format(
        chat_history=chat_hist_text,
        context=context,
        question=user_query
    )

    # Query Gemini
    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"⚠️ Gemini Error: {str(e)}"

    # Update chat history
    chat_history.append(("You", user_query))
    chat_history.append(("AI", answer))

    return answer, chat_history
