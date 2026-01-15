"""
Gradio Web UI for Belgian Bilingual Corporate Agent.

A user-friendly chat interface for querying Belgian banking reports with:
- Multilingual support (FR/NL/EN)
- Source attribution
- Bilingual consistency checking
"""
import gradio as gr
import sys
import requests
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Lazy loading - don't import RAGService until needed
rag_service = None
status_message = "RAG service will initialize on first query..."

# Global state for sources/metadata display
last_sources_html = ""
last_metadata_html = ""


def get_rag_service():
    """Lazy initialization of RAG service."""
    global rag_service, status_message

    if rag_service is not None:
        return rag_service, status_message

    print("Initializing RAG service (this may take 30-60 seconds on first load)...")

    try:
        from rag_service import RAGService
        rag_service = RAGService()
        info = rag_service.collection_info()

        if info['count'] == 0:
            status_message = "No documents indexed. Please run: python scripts/rag_cli.py index"
            return None, status_message

        # Check Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                status_message = f"Ollama not detected. Use 'Retrieval Only' mode. {info['count']} chunks indexed."
        except:
            status_message = f"Ollama not detected. Use 'Retrieval Only' mode. {info['count']} chunks indexed."
        else:
            status_message = f"System ready! {info['count']} chunks indexed."

        print(f"RAG service initialized: {status_message}")
        return rag_service, status_message

    except Exception as e:
        status_message = f"Error initializing: {str(e)}"
        return None, status_message


def format_sources_html(sources, docs):
    """Format retrieved sources as expandable HTML cards."""
    if not sources:
        return "<p>No sources retrieved.</p>"

    html = "<div style='font-family: sans-serif;'>"

    for i, (meta, doc) in enumerate(zip(sources, docs), 1):
        bank = meta.get('bank', 'Unknown')
        page = meta.get('page', '?')
        lang = meta.get('language', '?').upper()

        # Truncate long documents
        preview = doc[:300] + "..." if len(doc) > 300 else doc

        html += f"""
        <details style="margin: 10px 0; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #f9f9f9;">
            <summary style="cursor: pointer; font-weight: bold; color: #333;">
                Source {i}: {bank} (Page {page}, {lang})
            </summary>
            <div style="margin-top: 10px; padding: 10px; background: white; border-radius: 5px; font-size: 0.9em; line-height: 1.5;">
                {preview}
            </div>
        </details>
        """

    html += "</div>"
    return html


def format_metadata_html(detected_lang, consistency):
    """Format bilingual metadata as HTML."""
    if not detected_lang:
        return "<p style='color: #666;'>Bilingual validation not enabled.</p>"

    html = "<div style='font-family: sans-serif; padding: 10px; background: #f5f5f5; border-radius: 5px;'>"

    # Language badge
    html += f"""
    <div style='margin-bottom: 10px;'>
        <strong>Language:</strong>
        <span style='display: inline-block; padding: 2px 8px; border-radius: 12px; background: #e3f2fd; color: #1976d2; font-size: 0.9em;'>
            {detected_lang.upper()}
        </span>
    </div>
    """

    # Consistency status
    if consistency:
        status = consistency.get('status', 'unknown')
        confidence = consistency.get('confidence', 0.0)

        if status == 'ok':
            status_color = '#4caf50'
            status_text = 'OK'
        elif status == 'warning':
            status_color = '#ff9800'
            status_text = 'WARNING'
        else:
            status_color = '#f44336'
            status_text = 'ERROR'

        html += f"""
        <div style='margin-bottom: 10px;'>
            <strong>Consistency:</strong>
            <span style='color: {status_color}; font-weight: bold;'>
                {status_text}
            </span>
            <span style='color: #666; font-size: 0.9em;'>
                (confidence: {confidence:.2f})
            </span>
        </div>
        """

        if 'common_numbers' in consistency:
            count = len(consistency['common_numbers'])
            html += f"<div style='font-size: 0.9em; color: #666;'>Found {count} common numeric values across languages</div>"

    html += "</div>"
    return html


def respond(message, history, enable_bilingual, n_results, retrieval_only):
    """
    Process user message and return response.
    Uses generator to allow streaming-like updates.
    """
    global last_sources_html, last_metadata_html

    # Handle message being a list (Gradio 6 format) or string
    if isinstance(message, list):
        # Extract text from list of content blocks
        text_parts = []
        for part in message:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
        message = ' '.join(text_parts)

    if not message or not str(message).strip():
        return ""

    message = str(message).strip()

    # Lazy initialize RAG service
    rag, status = get_rag_service()

    if not rag:
        return f"Error: RAG service not initialized - {status}"

    try:
        # Query the RAG system
        result = rag.answer_question(
            question=message,
            n_results=int(n_results),
            use_llm=(not retrieval_only),
            enable_bilingual=enable_bilingual
        )

        # Check for errors
        answer = result.get('answer', 'No answer generated')
        if answer.startswith("Error:"):
            return answer

        # Update sources display
        sources = result.get('sources', [])
        docs = result.get('documents', [])
        if not docs:
            docs = [""] * len(sources)
        last_sources_html = format_sources_html(sources, docs)

        # Update metadata display
        detected_lang = result.get('language')
        consistency = result.get('consistency')
        last_metadata_html = format_metadata_html(detected_lang, consistency)

        # Format response
        if retrieval_only:
            return f"**Retrieved {result.get('num_sources', 0)} sources** (LLM disabled)\n\nSee 'Retrieved Sources' panel for details."
        else:
            return answer

    except Exception as e:
        return f"Error: {str(e)}"


def get_sources():
    """Return current sources HTML."""
    return last_sources_html


def get_metadata():
    """Return current metadata HTML."""
    return last_metadata_html


# Build Gradio interface
with gr.Blocks(title="Belgian Banking RAG") as demo:
    gr.Markdown("# Belgian Bilingual Corporate Agent")
    gr.Markdown("Ask questions about **BNP Paribas Fortis** and **KBC Group** annual reports in English, French, or Dutch.")
    gr.Markdown(f"*Status: {status_message}*")

    with gr.Row():
        # Left column: Chat
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=450
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a question...",
                    label="Your Question",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("Clear Chat")

            gr.Markdown("### Example Questions")
            with gr.Row():
                ex1 = gr.Button("What is KBC's CET1 ratio?", size="sm")
                ex2 = gr.Button("Net profit of BNP Paribas Fortis?", size="sm")

        # Right column: Settings and info
        with gr.Column(scale=1):
            with gr.Accordion("Settings", open=True):
                bilingual = gr.Checkbox(
                    label="Enable Bilingual Validation",
                    value=False,
                    info="Cross-reference FR/NL versions"
                )
                n_results = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Number of Sources"
                )
                retrieval_only = gr.Checkbox(
                    label="Retrieval Only (Skip LLM)",
                    value=False,
                    info="Faster - no answer generation"
                )

            with gr.Accordion("Response Metadata", open=True):
                metadata_display = gr.HTML(
                    value="<p style='color: #666;'>Ask a question with bilingual validation enabled.</p>"
                )

            with gr.Accordion("Retrieved Sources", open=True):
                sources_display = gr.HTML(
                    value="<p style='color: #666;'>Sources will appear here.</p>"
                )

    # Chat function that returns proper message format
    def user_message(message, history):
        """Add user message to history."""
        if not message.strip():
            return "", history
        return "", history + [{"role": "user", "content": message}]

    def bot_response(history, enable_bilingual, n_results, retrieval_only):
        """Generate bot response."""
        if not history:
            return history, "", ""

        # Get the last user message
        last_msg = history[-1]
        if last_msg.get("role") != "user":
            return history, last_sources_html, last_metadata_html

        # Extract text content (can be string or list in Gradio 6)
        content = last_msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and 'text' in part:
                    text_parts.append(part['text'])
            user_text = ' '.join(text_parts)
        else:
            user_text = str(content)

        # Generate response
        response = respond(user_text, history[:-1], enable_bilingual, n_results, retrieval_only)

        # Add assistant message
        history.append({"role": "assistant", "content": response})

        return history, last_sources_html, last_metadata_html

    # Wire up events
    msg.submit(
        user_message,
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        bot_response,
        [chatbot, bilingual, n_results, retrieval_only],
        [chatbot, sources_display, metadata_display]
    )

    submit_btn.click(
        user_message,
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        bot_response,
        [chatbot, bilingual, n_results, retrieval_only],
        [chatbot, sources_display, metadata_display]
    )

    clear_btn.click(
        lambda: ([], "<p style='color: #666;'>Sources will appear here.</p>", "<p style='color: #666;'>Ask a question with bilingual validation enabled.</p>"),
        outputs=[chatbot, sources_display, metadata_display]
    )

    # Example buttons
    ex1.click(lambda: "What is KBC's Common Equity Tier 1 ratio?", outputs=[msg])
    ex2.click(lambda: "What is the net profit of BNP Paribas Fortis?", outputs=[msg])


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Belgian Bilingual Corporate Agent - Gradio UI")
    print("="*60)
    print(f"\nStatus: {status_message}")
    print("\nStarting server (will find available port)...")
    print("Press Ctrl+C to stop.")
    print("="*60 + "\n")

    demo.launch(
        server_name="127.0.0.1",
        share=False
    )
