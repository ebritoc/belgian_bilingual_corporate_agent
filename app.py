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

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Lazy loading - don't import RAGService until needed
rag_service = None
app_config = None
status_message = "RAG service will initialize on first query..."

# Global state for sources/metadata display
last_sources_html = ""
last_metadata_html = ""
last_consistency_html = ""
last_passages = []  # Store passages for visualization


def load_config() -> dict:
    """Load configuration from YAML file."""
    global app_config
    if app_config is not None:
        return app_config

    config_path = Path(__file__).parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            app_config = yaml.safe_load(f)
    else:
        app_config = {}
    return app_config


def get_rag_service():
    """Lazy initialization of RAG service."""
    global rag_service, status_message

    if rag_service is not None:
        return rag_service, status_message

    print("Initializing RAG service (this may take 30-60 seconds on first load)...")

    try:
        # Load config
        config = load_config()

        # Build service config from YAML
        service_config = {}

        # Retriever settings
        retriever_config = config.get('retriever', {})
        service_config['retriever_type'] = retriever_config.get('type', 'chroma')
        service_config.update(retriever_config)

        # LLM settings
        llm_config = config.get('llm', {})
        service_config['ollama_url'] = llm_config.get('ollama_url', 'http://localhost:11434')
        service_config['model_name'] = llm_config.get('model_name', 'qwen2.5:7b')
        service_config['ollama_timeout'] = llm_config.get('timeout', 180)

        from rag_service import RAGService
        rag_service = RAGService(service_config)
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


def format_consistency_viz_html(passages, token_consistency=None):
    """Format consistency visualization as HTML with embedded info."""
    if not passages:
        return "<p style='color: #666;'>No passages with embeddings available for visualization.</p>"

    # Check if any passage has embeddings (ColBERT retriever)
    has_embeddings = any(
        getattr(p, 'query_embeddings', None) is not None
        for p in passages
    )

    if not has_embeddings:
        return """
        <div style='padding: 15px; background: #fff3e0; border-radius: 8px; border-left: 4px solid #ff9800;'>
            <strong>Token-level explanations require ColBERT retriever</strong>
            <p style='margin: 5px 0 0 0; font-size: 0.9em; color: #666;'>
                Switch to ColBERT in config.yaml to enable MaxSimE visualizations.
            </p>
        </div>
        """

    # If we have token alignment consistency results
    if token_consistency and token_consistency.get('results'):
        results = token_consistency['results']
        status = token_consistency.get('status', 'ok')
        notes = token_consistency.get('notes', '')

        if status == 'ok':
            status_color = '#4CAF50'
            icon = '&#10003;'  # checkmark
        elif status == 'warning':
            status_color = '#FF9800'
            icon = '&#9888;'  # warning
        else:
            status_color = '#F44336'
            icon = '&#10007;'  # x mark

        html = f"""
        <div style='padding: 15px; background: #f5f5f5; border-radius: 8px; border-left: 4px solid {status_color};'>
            <div style='font-size: 1.1em; font-weight: bold; color: {status_color}; margin-bottom: 10px;'>
                {icon} Token Alignment: {status.upper()}
            </div>
            <p style='margin: 0 0 10px 0; font-size: 0.9em; color: #666;'>{notes}</p>
        """

        # Show individual results
        for i, result in enumerate(results[:3]):  # Limit to 3
            score = result.score
            overlap = result.alignment_overlap
            flagged = result.flagged

            bar_color = '#F44336' if flagged else '#4CAF50'
            html += f"""
            <div style='margin: 8px 0; padding: 8px; background: white; border-radius: 4px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span>Pair {i+1}</span>
                    <span style='color: {bar_color}; font-weight: bold;'>
                        Score: {score:.2f} | Overlap: {overlap:.0%}
                    </span>
                </div>
                <div style='margin-top: 5px; height: 6px; background: #eee; border-radius: 3px;'>
                    <div style='height: 100%; width: {score*100}%; background: {bar_color}; border-radius: 3px;'></div>
                </div>
            </div>
            """

        html += "</div>"
        return html

    return """
    <div style='padding: 15px; background: #e3f2fd; border-radius: 8px; border-left: 4px solid #2196F3;'>
        <strong>Embeddings available</strong>
        <p style='margin: 5px 0 0 0; font-size: 0.9em; color: #666;'>
            Enable bilingual validation to see token alignment analysis.
        </p>
    </div>
    """


def format_metadata_html(detected_lang, consistency, source_langs=None):
    """Format bilingual metadata as HTML."""
    if not detected_lang and not source_langs:
        return "<p style='color: #666;'>Bilingual validation not enabled.</p>"

    html = "<div style='font-family: sans-serif; padding: 10px; background: #f5f5f5; border-radius: 5px;'>"

    # Query language badge
    if detected_lang:
        html += f"""
    <div style='margin-bottom: 10px;'>
        <strong>Query:</strong>
        <span style='display: inline-block; padding: 2px 8px; border-radius: 12px; background: #e3f2fd; color: #1976d2; font-size: 0.9em;'>
            {detected_lang.upper()}
        </span>
    </div>
    """

    # Source language breakdown
    if source_langs:
        from collections import Counter
        lang_counts = Counter(source_langs)
        badges = " ".join(
            f"<span style='display: inline-block; padding: 2px 8px; border-radius: 12px; "
            f"background: #e8f5e9; color: #2e7d32; font-size: 0.9em; margin-right: 4px;'>"
            f"{lang.upper()} &times;{count}</span>"
            for lang, count in sorted(lang_counts.items())
        )
        html += f"<div style='margin-bottom: 10px;'><strong>Sources:</strong> {badges}</div>"

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
    global last_sources_html, last_metadata_html, last_consistency_html, last_passages

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

        # Store passages for visualization (if available from ColBERT)
        last_passages = result.get('passages', [])

        # Update metadata display
        detected_lang = result.get('language')
        consistency = result.get('consistency')
        source_langs = [s.get('language') for s in sources if s.get('language')]
        last_metadata_html = format_metadata_html(detected_lang, consistency, source_langs)

        # Update consistency visualization
        token_consistency = result.get('token_consistency')
        last_consistency_html = format_consistency_viz_html(last_passages, token_consistency)

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

            with gr.Accordion("Token Alignment", open=False):
                consistency_display = gr.HTML(
                    value="<p style='color: #666;'>Token-level consistency analysis will appear here when using ColBERT retriever with bilingual validation.</p>"
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
            return history, "", "", ""

        # Get the last user message
        last_msg = history[-1]
        if last_msg.get("role") != "user":
            return history, last_sources_html, last_metadata_html, last_consistency_html

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

        return history, last_sources_html, last_metadata_html, last_consistency_html

    # Wire up events
    msg.submit(
        user_message,
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        bot_response,
        [chatbot, bilingual, n_results, retrieval_only],
        [chatbot, sources_display, metadata_display, consistency_display]
    )

    submit_btn.click(
        user_message,
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        bot_response,
        [chatbot, bilingual, n_results, retrieval_only],
        [chatbot, sources_display, metadata_display, consistency_display]
    )

    clear_btn.click(
        lambda: (
            [],
            "<p style='color: #666;'>Sources will appear here.</p>",
            "<p style='color: #666;'>Ask a question with bilingual validation enabled.</p>",
            "<p style='color: #666;'>Token-level consistency analysis will appear here.</p>"
        ),
        outputs=[chatbot, sources_display, metadata_display, consistency_display]
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
