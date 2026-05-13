"""Generate a PDF security report using ReportLab."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .charts import plot_attack_success_bar, plot_detection_confusion
from .templates import build_recommendations, format_model_details


def _table(title: str, data: Sequence[Sequence[object]]) -> Table:
    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ]
        )
    )
    return t


def build_pdf(
    output_path: Path,
    project_title: str,
    model_info: Mapping[str, object],
    attack_results: Sequence[Mapping[str, object]],
    detection_results: Iterable[Mapping[str, object]],
    summary: Mapping[str, float],
    diagnostics: Mapping[str, object] = None,  # NEW: Optional diagnostics
) -> Path:
    """Create a PDF report at output_path.

    Args:
        output_path: Destination PDF path.
        project_title: Title to display.
        model_info: Dict with name/version/framework/notes.
        attack_results: Sequence of attack outputs (from AttackRunner).
        detection_results: Iterable of detection outputs.
        summary: Aggregated metrics dict from analysis.summarizer.summarize.
        diagnostics: Optional diagnostic analysis results with model health and issues.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm)
    elems = []

    elems.append(Paragraph(project_title, styles["Title"]))
    elems.append(Spacer(1, 0.4 * cm))

    info = format_model_details(model_info)
    elems.append(Paragraph("Model Details", styles["Heading2"]))
    elems.append(Paragraph(f"Name: {info['name']}", styles["BodyText"]))
    elems.append(Paragraph(f"Version: {info['version']}", styles["BodyText"]))
    elems.append(Paragraph(f"Framework: {info['framework']}", styles["BodyText"]))
    if info.get("notes"):
        elems.append(Paragraph(f"Notes: {info['notes']}", styles["BodyText"]))
    elems.append(Spacer(1, 0.3 * cm))

    elems.append(Paragraph("Summary Metrics", styles["Heading2"]))
    metrics_table = _table(
        "Summary",
        [
            ["Metric", "Value"],
            ["Attack Success Rate", f"{summary.get('attack_success_rate', 0):.3f}"],
            ["Detection Accuracy", f"{summary.get('detection_accuracy', 0):.3f}"],
            ["False Positive Rate", f"{summary.get('false_positive_rate', 0):.3f}"],
            ["Robustness Score", f"{summary.get('robustness_score', 0):.1f}/100"],
        ],
    )
    elems.append(metrics_table)
    elems.append(Spacer(1, 0.4 * cm))

    # Charts (persist so ReportLab can read them during build)
    charts_dir = output_path.parent / "_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    attack_chart = plot_attack_success_bar(attack_results, charts_dir / "attack_rates.png")
    det_chart = plot_detection_confusion(detection_results, charts_dir / "det_confusion.png")

    if attack_chart and attack_chart.exists():
        elems.append(Paragraph("Attack Success Rates", styles["Heading2"]))
        elems.append(Image(str(attack_chart), width=12 * cm, height=6 * cm))
        elems.append(Spacer(1, 0.3 * cm))
    if det_chart and det_chart.exists():
        elems.append(Paragraph("Detection Confusion", styles["Heading2"]))
        elems.append(Image(str(det_chart), width=8 * cm, height=8 * cm))
        elems.append(Spacer(1, 0.3 * cm))

    # NEW: Diagnostic Insights Section
    if diagnostics and diagnostics.get('diagnostics'):
        elems.append(Spacer(1, 0.5 * cm))
        elems.append(Paragraph("Diagnostic Insights", styles["Heading1"]))
        elems.append(Spacer(1, 0.2 * cm))

        # Overall health banner
        health = diagnostics.get('overall_health', 'UNKNOWN')
        health_colors_map = {
            'HEALTHY': colors.green,
            'CONCERNING': colors.orange,
            'CRITICAL': colors.red
        }
        health_color = health_colors_map.get(health, colors.grey)

        health_para = Paragraph(
            f"<b>Model Health Status: </b><font color='{health_color.hexval()}'><b>{health}</b></font>",
            styles["BodyText"]
        )
        elems.append(health_para)
        elems.append(Spacer(1, 0.3 * cm))

        # Show LLM enhancement indicator if enabled
        if diagnostics.get('llm_enhanced'):
            llm_indicator = Paragraph(
                "<i>✨ Enhanced with AI-powered analysis</i>",
                styles["BodyText"]
            )
            elems.append(llm_indicator)
            elems.append(Spacer(1, 0.2 * cm))

        # List each diagnostic issue
        diag_list = diagnostics.get('diagnostics', [])
        if diag_list:
            for i, diag in enumerate(diag_list, 1):
                severity = diag.get('severity', 'INFO')
                description = diag.get('description', 'No description')
                sev_color = _get_severity_color(severity)

                # Issue header with severity
                header_text = f"<b>{i}. [{severity}]</b> {description}"
                header_para = Paragraph(header_text, styles["Heading3"])
                # Note: ReportLab doesn't support textColor directly in paragraph style,
                # so we use font color tags in the text
                elems.append(header_para)

                # Evidence section
                evidence = diag.get('evidence')
                if evidence and isinstance(evidence, Mapping):
                    elems.append(Paragraph("<b>Evidence:</b>", styles["Heading4"]))
                    evidence_table = _format_evidence_table(evidence)
                    elems.append(evidence_table)
                    elems.append(Spacer(1, 0.2 * cm))

                # Interpretation/Explanation
                interpretation = diag.get('interpretation') or diag.get('explanation')
                if interpretation:
                    elems.append(Paragraph(f"<i>{interpretation}</i>", styles["BodyText"]))
                    elems.append(Spacer(1, 0.2 * cm))

                # Root causes (if available)
                root_causes = diag.get('root_causes')
                if root_causes and isinstance(root_causes, (list, tuple)):
                    elems.append(Paragraph("<b>Possible Root Causes:</b>", styles["Heading4"]))
                    for cause in root_causes:
                        elems.append(Paragraph(f"- {cause}", styles["BodyText"]))
                    elems.append(Spacer(1, 0.2 * cm))

                # Recommendations
                recommendations = diag.get('recommendations')
                if recommendations and isinstance(recommendations, (list, tuple)):
                    elems.append(Paragraph("<b>Recommendations:</b>", styles["Heading4"]))
                    for rec in recommendations:
                        elems.append(Paragraph(f"- {rec}", styles["BodyText"]))
                    elems.append(Spacer(1, 0.2 * cm))

                # Verdict (if available)
                verdict = diag.get('verdict')
                if verdict:
                    elems.append(Paragraph(f"<b>Verdict:</b> {verdict}", styles["BodyText"]))

                elems.append(Spacer(1, 0.4 * cm))
        else:
            elems.append(Paragraph("No critical issues detected. Model appears to be functioning correctly.", styles["BodyText"]))
            elems.append(Spacer(1, 0.3 * cm))

    # Attacks performed table
    if attack_results:
        rows = [["Attack", "Success Rate"]]
        for entry in attack_results:
            name = entry.get("attack") if isinstance(entry, Mapping) else "Attack"
            res = entry.get("result") if isinstance(entry, Mapping) else None
            rate = "n/a"
            if isinstance(res, Mapping) and isinstance(res.get("attack_success_rate"), (float, int)):
                rate = f"{float(res['attack_success_rate']):.3f}"
            rows.append([str(name), rate])
        elems.append(Paragraph("Attacks Performed", styles["Heading2"]))
        elems.append(_table("Attacks", rows))
        elems.append(Spacer(1, 0.3 * cm))

    # Detection results table
    det_rows = [["is_attack", "detected"]]
    added = False
    for entry in detection_results:
        if not isinstance(entry, Mapping):
            continue
        ia = entry.get("is_attack")
        det = entry.get("detected")
        if isinstance(ia, bool) and isinstance(det, bool):
            det_rows.append([str(ia), str(det)])
            added = True
    if added:
        elems.append(Paragraph("Detection Samples", styles["Heading2"]))
        elems.append(_table("Detection", det_rows))
        elems.append(Spacer(1, 0.3 * cm))

    # Recommendations
    elems.append(Paragraph("Security Recommendations", styles["Heading2"]))
    for rec in build_recommendations(summary):
        elems.append(Paragraph(f"- {rec}", styles["BodyText"]))

    doc.build(elems)
    return output_path


def _get_severity_color(severity: str) -> colors.Color:
    """Map severity level to color.

    Args:
        severity: Severity string (CRITICAL, HIGH, MEDIUM, WARNING, LOW, INFO)

    Returns:
        ReportLab color for the severity
    """
    severity_colors = {
        'CRITICAL': colors.red,
        'HIGH': colors.orangered,
        'MEDIUM': colors.orange,
        'WARNING': colors.yellow,
        'LOW': colors.yellow,
        'INFO': colors.blue
    }
    return severity_colors.get(severity, colors.grey)


def _format_evidence_table(evidence: Mapping[str, object]) -> Table:
    """Format evidence dict as a table.

    Args:
        evidence: Dictionary of evidence metrics

    Returns:
        ReportLab Table with evidence data
    """
    data = [["Metric", "Value"]]

    for key, value in evidence.items():
        # Special handling for sample_mismatches (nested list of dicts)
        if key == 'sample_mismatches' and isinstance(value, list):
            if len(value) == 0:
                value_str = "No mismatches"
            else:
                # Format each mismatch as readable text
                mismatch_lines = []
                for i, mismatch in enumerate(value[:5], 1):  # Show first 5
                    sample_id = mismatch.get('sample_id', 'Unknown')
                    # Truncate long sample IDs
                    if len(sample_id) > 40:
                        sample_id = sample_id[:40] + "..."

                    gt = mismatch.get('ground_truth', '?')
                    pred = mismatch.get('predicted', '?')
                    conf = mismatch.get('confidence', 0.0)

                    mismatch_lines.append(
                        f"{i}. '{sample_id}'\n"
                        f"   GT: {gt} | Pred: {pred} | Conf: {conf:.2f}"
                    )

                value_str = "\n\n".join(mismatch_lines)
                if len(value) > 5:
                    value_str += f"\n\n... and {len(value) - 5} more"
        # Format value based on type
        elif isinstance(value, float):
            value_str = f"{value:.4f}"
        elif isinstance(value, dict):
            # Truncate long dicts
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:50] + "..."
        elif isinstance(value, list):
            # Show first few items
            if len(value) <= 3:
                value_str = str(value)
            else:
                value_str = f"[{', '.join(str(v) for v in value[:3])}... ({len(value)} total)]"
        else:
            value_str = str(value)

        # Format key (replace underscores, title case)
        key_formatted = key.replace('_', ' ').title()

        data.append([key_formatted, value_str])

    table = Table(data, colWidths=[6 * cm, 9 * cm])
    table.setStyle(
        TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ])
    )
    return table
