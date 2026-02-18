"""
Visualization module for MaxSimE explanations in Gradio.

Wraps MaxSimE plotting functions and provides crosslingual consistency
comparison views for the Gradio UI.
"""
from typing import Optional, List, Tuple, TYPE_CHECKING
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Gradio
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from consistency import ConsistencyResult

try:
    from maxsime import (
        MaxSimExplanation,
        plot_similarity_heatmap,
        plot_token_alignment,
    )
    MAXSIME_AVAILABLE = True
except ImportError:
    MAXSIME_AVAILABLE = False
    MaxSimExplanation = None


def render_explanation_heatmap(
    explanation: "MaxSimExplanation",
    title: str = "",
    figsize: Tuple[int, int] = (12, 5),
    max_doc_tokens: int = 80,
) -> plt.Figure:
    """
    Render a single query-doc heatmap for Gradio.

    Args:
        explanation: MaxSimExplanation object
        title: Plot title
        figsize: Figure size (width, height)
        max_doc_tokens: Maximum document tokens to display

    Returns:
        matplotlib Figure
    """
    if not MAXSIME_AVAILABLE:
        return _placeholder_figure("MaxSimE not installed")

    return plot_similarity_heatmap(
        explanation,
        title=title,
        figsize=figsize,
        max_doc_tokens=max_doc_tokens,
    )


def render_token_alignment(
    explanation: "MaxSimExplanation",
    top_k: int = 10,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Render top-k token alignment diagram for Gradio.

    Args:
        explanation: MaxSimExplanation object
        top_k: Number of top alignments to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if not MAXSIME_AVAILABLE:
        return _placeholder_figure("MaxSimE not installed")

    return plot_token_alignment(
        explanation,
        top_k=top_k,
        figsize=figsize,
    )


def render_consistency_comparison(
    result: "ConsistencyResult",
    query_fr: str = "FR Query",
    query_nl: str = "NL Query",
    title: str = "",
    figsize: Tuple[int, int] = (16, 10),
    max_doc_tokens: int = 60,
) -> plt.Figure:
    """
    Render side-by-side comparison of FR and NL query alignments.

    Layout:
    ┌──────────────────┬──────────────────┐
    │  FR query → doc  │  NL query → doc  │
    │   (heatmap)      │   (heatmap)      │
    ├──────────────────┴──────────────────┤
    │  Doc token overlap visualization    │
    │  ■ FR only  ■ NL only  ■ Both      │
    ├─────────────────────────────────────┤
    │  Score summary                      │
    └─────────────────────────────────────┘

    Args:
        result: ConsistencyResult with explanations
        query_fr: FR query text for label
        query_nl: NL query text for label
        title: Overall title
        figsize: Figure size
        max_doc_tokens: Max tokens per heatmap

    Returns:
        matplotlib Figure
    """
    if not MAXSIME_AVAILABLE:
        return _placeholder_figure("MaxSimE not installed")

    if result.explanation_fr is None or result.explanation_nl is None:
        return _placeholder_figure("Explanations not available")

    fig = plt.figure(figsize=figsize)

    # Create grid: 2 columns for heatmaps, 1 row for overlap bar, 1 row for summary
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 1, 0.5], hspace=0.3, wspace=0.2)

    # Top left: FR heatmap
    ax_fr = fig.add_subplot(gs[0, 0])
    _render_mini_heatmap(ax_fr, result.explanation_fr, f"FR: {query_fr[:40]}...", max_doc_tokens)

    # Top right: NL heatmap
    ax_nl = fig.add_subplot(gs[0, 1])
    _render_mini_heatmap(ax_nl, result.explanation_nl, f"NL: {query_nl[:40]}...", max_doc_tokens)

    # Middle: Token overlap bar
    ax_overlap = fig.add_subplot(gs[1, :])
    _render_overlap_bar(ax_overlap, result, max_doc_tokens)

    # Bottom: Score summary
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')

    icon = "✓" if not result.flagged else "✗"
    status = "CONSISTENT" if not result.flagged else "INCONSISTENT"
    color = "#4CAF50" if not result.flagged else "#F44336"

    summary_text = (
        f"{icon} {status}  |  "
        f"Score: {result.score:.2f}  |  "
        f"Overlap: {result.alignment_overlap:.1%}  |  "
        f"Divergence: {result.score_divergence:.1%}"
    )
    ax_summary.text(
        0.5, 0.5, summary_text,
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        color=color,
        transform=ax_summary.transAxes,
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def _render_mini_heatmap(
    ax: plt.Axes,
    explanation: "MaxSimExplanation",
    title: str,
    max_tokens: int,
) -> None:
    """Render a simplified heatmap on the given axes."""
    sim_matrix = explanation.similarity_matrix

    # Truncate if needed
    if sim_matrix.shape[1] > max_tokens:
        sim_matrix = sim_matrix[:, :max_tokens]

    # Plot heatmap
    im = ax.imshow(sim_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    # Labels
    q_tokens = explanation.query_tokens[:sim_matrix.shape[0]]
    d_tokens = explanation.doc_tokens[:sim_matrix.shape[1]]

    # Truncate long tokens
    q_labels = [t[:8] for t in q_tokens]
    d_labels = [t[:6] if i % 3 == 0 else '' for i, t in enumerate(d_tokens)]

    ax.set_yticks(range(len(q_labels)))
    ax.set_yticklabels(q_labels, fontsize=8)
    ax.set_xticks(range(len(d_labels)))
    ax.set_xticklabels(d_labels, fontsize=6, rotation=45, ha='right')

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Document tokens', fontsize=8)
    ax.set_ylabel('Query tokens', fontsize=8)


def _render_overlap_bar(
    ax: plt.Axes,
    result: "ConsistencyResult",
    max_tokens: int,
) -> None:
    """Render a bar showing which doc tokens are matched by FR, NL, or both."""
    # Determine total tokens to display
    all_matched = result.matched_doc_tokens_fr | result.matched_doc_tokens_nl
    if not all_matched:
        ax.text(0.5, 0.5, 'No matched tokens', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return

    max_idx = min(max(all_matched) + 1, max_tokens)

    # Create color array
    colors = []
    for i in range(max_idx):
        in_fr = i in result.matched_doc_tokens_fr
        in_nl = i in result.matched_doc_tokens_nl
        if in_fr and in_nl:
            colors.append('#4CAF50')  # Green - both
        elif in_fr:
            colors.append('#2196F3')  # Blue - FR only
        elif in_nl:
            colors.append('#FF9800')  # Orange - NL only
        else:
            colors.append('#EEEEEE')  # Gray - neither

    # Plot bars
    ax.bar(range(max_idx), [1] * max_idx, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xlim(-0.5, max_idx - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Document token index', fontsize=9)
    ax.set_yticks([])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Both'),
        Patch(facecolor='#2196F3', label='FR only'),
        Patch(facecolor='#FF9800', label='NL only'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=3)

    ax.set_title('Document Token Coverage', fontsize=10)


def format_consistency_html(
    result: "ConsistencyResult",
    doc_tokens: Optional[List[str]] = None,
) -> str:
    """
    Format consistency result as HTML for Gradio display.

    Args:
        result: ConsistencyResult to format
        doc_tokens: Optional document token strings

    Returns:
        HTML string with styled status
    """
    if result.flagged:
        icon = "🔴"
        status = "Inconsistent"
        status_color = "#F44336"
    else:
        icon = "🟢"
        status = "Consistent"
        status_color = "#4CAF50"

    html = f"""
    <div style="font-family: sans-serif; padding: 15px; background: #f5f5f5; border-radius: 8px; border-left: 4px solid {status_color};">
        <div style="font-size: 1.2em; font-weight: bold; color: {status_color}; margin-bottom: 10px;">
            {icon} {status}
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 10px;">
            <div style="background: white; padding: 10px; border-radius: 5px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold;">{result.score:.2f}</div>
                <div style="font-size: 0.8em; color: #666;">Score</div>
            </div>
            <div style="background: white; padding: 10px; border-radius: 5px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold;">{result.alignment_overlap:.0%}</div>
                <div style="font-size: 0.8em; color: #666;">Overlap</div>
            </div>
            <div style="background: white; padding: 10px; border-radius: 5px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold;">{result.score_divergence:.0%}</div>
                <div style="font-size: 0.8em; color: #666;">Divergence</div>
            </div>
        </div>
    """

    if doc_tokens:
        # Show matched tokens
        fr_tokens = [doc_tokens[i] for i in sorted(result.matched_doc_tokens_fr) if i < len(doc_tokens)][:8]
        nl_tokens = [doc_tokens[i] for i in sorted(result.matched_doc_tokens_nl) if i < len(doc_tokens)][:8]
        overlap = [doc_tokens[i] for i in sorted(result.overlap_tokens) if i < len(doc_tokens)][:8]

        html += f"""
        <div style="font-size: 0.9em; margin-top: 10px;">
            <div><strong style="color: #2196F3;">FR matched:</strong> {', '.join(fr_tokens)}{'...' if len(result.matched_doc_tokens_fr) > 8 else ''}</div>
            <div><strong style="color: #FF9800;">NL matched:</strong> {', '.join(nl_tokens)}{'...' if len(result.matched_doc_tokens_nl) > 8 else ''}</div>
            <div><strong style="color: #4CAF50;">Shared:</strong> {', '.join(overlap)}{'...' if len(result.overlap_tokens) > 8 else ''}</div>
        </div>
        """

    html += "</div>"
    return html


def _placeholder_figure(message: str) -> plt.Figure:
    """Create a placeholder figure with a message."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5, 0.5, message,
        ha='center', va='center',
        fontsize=14, color='gray',
        transform=ax.transAxes,
    )
    ax.axis('off')
    return fig
