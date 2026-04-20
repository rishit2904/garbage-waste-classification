"""
Generate a clean, formatted PDF of all project source code.
Designed for academic submission on Teams.
"""

import os
from fpdf import FPDF
from datetime import datetime


class CodePDF(FPDF):
    """Custom PDF class with header/footer for code documentation."""

    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.project_title = "Garbage Waste Classification System"
        self.subtitle = "Deep Learning Mini Project - Source Code"

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, self.project_title, align='L')
        self.cell(0, 6, f'Page {self.page_no()}', align='R', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%d %B %Y, %I:%M %p")}', align='C')


def add_cover_page(pdf):
    """Add a professional cover page."""
    pdf.add_page()
    pdf.ln(60)

    # Title
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(26, 86, 219)
    pdf.cell(0, 14, "Garbage Waste Classification", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 14, "System", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(6)

    # Subtitle
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Deep Learning Mini Project", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(4)

    # Divider
    pdf.set_draw_color(26, 86, 219)
    pdf.set_line_width(0.6)
    pdf.line(70, pdf.get_y(), 140, pdf.get_y())
    pdf.ln(10)

    # Description
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "AI-Powered Waste Classification for Smarter Recycling", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 8, "CNN with Transfer Learning (ResNet18)", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(8)

    # Tech stack
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, "Python  |  PyTorch  |  Flask  |  Albumentations  |  scikit-learn", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(20)

    # Date
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 8, datetime.now().strftime("%B %Y"), align='C', new_x='LMARGIN', new_y='NEXT')


def add_table_of_contents(pdf, files):
    """Add a table of contents page."""
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, "Table of Contents", new_x='LMARGIN', new_y='NEXT')
    pdf.ln(4)

    pdf.set_draw_color(26, 86, 219)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 60, pdf.get_y())
    pdf.ln(8)

    for i, (filepath, section, description) in enumerate(files, 1):
        # Section number and name
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(26, 86, 219)
        pdf.cell(10, 8, f"{i}.")

        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(80, 8, section)

        # File path
        pdf.set_font('Courier', '', 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 8, filepath, new_x='LMARGIN', new_y='NEXT')

        # Description
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(10, 6, "")
        pdf.cell(0, 6, description, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(3)


def add_code_file(pdf, filepath, section_name, section_num, base_dir):
    """Add a source code file with syntax highlighting cues."""
    pdf.add_page()

    # Section header
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(26, 86, 219)
    pdf.cell(0, 10, f"{section_num}. {section_name}", new_x='LMARGIN', new_y='NEXT')

    # File path
    pdf.set_font('Courier', '', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, filepath, new_x='LMARGIN', new_y='NEXT')

    # Divider
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(6)

    # Read file content
    full_path = os.path.join(base_dir, filepath)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 8, f"Error reading file: {e}", new_x='LMARGIN', new_y='NEXT')
        return

    # Line count info
    lines = content.split('\n')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(0, 5, f"{len(lines)} lines", new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    # Code content with line numbers
    pdf.set_font('Courier', '', 7.2)
    line_num_width = 12

    for i, line in enumerate(lines, 1):
        # Check if we need a new page
        if pdf.get_y() > 272:
            pdf.add_page()
            # Mini header on continuation pages
            pdf.set_font('Helvetica', 'I', 8)
            pdf.set_text_color(140, 140, 140)
            pdf.cell(0, 5, f"{section_name} (continued)", new_x='LMARGIN', new_y='NEXT')
            pdf.ln(3)
            pdf.set_font('Courier', '', 7.2)

        # Line number
        pdf.set_text_color(180, 180, 180)
        pdf.cell(line_num_width, 3.8, str(i).rjust(4), align='R')

        # Separator
        pdf.set_text_color(210, 210, 210)
        pdf.cell(3, 3.8, "|")

        # Code line - handle special characters  
        safe_line = line.replace('\t', '    ')
        # Truncate very long lines to fit page
        max_chars = 115
        if len(safe_line) > max_chars:
            safe_line = safe_line[:max_chars] + " ..."

        # Determine color based on content
        stripped = safe_line.strip()
        if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            # Comments
            pdf.set_text_color(100, 140, 100)
        elif stripped.startswith('def ') or stripped.startswith('class ') or stripped.startswith('function'):
            # Definitions
            pdf.set_text_color(26, 86, 219)
        elif stripped.startswith('import ') or stripped.startswith('from ') or stripped.startswith('@'):
            # Imports and decorators
            pdf.set_text_color(160, 100, 40)
        elif stripped.startswith('"""') or stripped.startswith("'''"):
            # Docstrings
            pdf.set_text_color(100, 140, 100)
        elif stripped.startswith('return '):
            # Return statements
            pdf.set_text_color(180, 60, 60)
        else:
            # Regular code
            pdf.set_text_color(40, 40, 40)

        pdf.cell(0, 3.8, safe_line, new_x='LMARGIN', new_y='NEXT')


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Files to include: (relative_path, section_name, description)
    files = [
        ("app/app.py", "Flask Application", "Web server, model inference, and API endpoints"),
        ("app/templates/index.html", "HTML Template", "Frontend UI with upload, classification display, and category cards"),
        ("app/static/style.css", "Stylesheet", "Flat light theme design system with solid colors"),
        ("model/cnn_model.py", "CNN Architecture", "Transfer learning (ResNet18), MobileNetV2, and custom CNN definitions"),
        ("model/train.py", "Training Pipeline", "Data preprocessing, augmentation, training loop, and evaluation"),
        ("model/augmentation.py", "Augmentation Pipeline", "Heavy data augmentation for real-world waste images"),
        ("requirements.txt", "Dependencies", "Python package requirements"),
    ]

    pdf = CodePDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover page
    add_cover_page(pdf)

    # Table of contents
    add_table_of_contents(pdf, files)

    # Code files
    for i, (filepath, section, desc) in enumerate(files, 1):
        add_code_file(pdf, filepath, section, i, base_dir)

    # Output
    output_path = os.path.join(base_dir, "Garbage_Waste_Classification_Code.pdf")
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")
    print(f"Pages: {pdf.pages_count}")


if __name__ == "__main__":
    main()
