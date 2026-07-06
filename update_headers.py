import os
import ast

header_block = """__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"
"""

dunder_vars = {"__author__", "__copyright__", "__credits__", "__license__", "__version__", "__maintainer__", "__email__", "__status__"}

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        print(f"Syntax error in {filepath}, skipping.")
        return

    # Find existing dunder assignments to remove them
    lines = content.split('\n')
    lines_to_keep = []
    
    for i, line in enumerate(lines):
        is_dunder = False
        for var in dunder_vars:
            if line.startswith(var) and "=" in line:
                is_dunder = True
                break
        if not is_dunder:
            lines_to_keep.append(line)
    
    new_content_str = '\n'.join(lines_to_keep)
    
    # Reparse to find insertion point
    try:
        tree = ast.parse(new_content_str)
    except SyntaxError:
        print(f"Syntax error after stripping dunders in {filepath}, skipping.")
        return

    last_import_line = 0
    docstring_end_line = 0
    
    if ast.get_docstring(tree, clean=False) is not None:
        if len(tree.body) > 0 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            docstring_end_line = tree.body[0].end_lineno
        # For python < 3.8
        elif len(tree.body) > 0 and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
            docstring_end_line = tree.body[0].end_lineno

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if node.end_lineno > last_import_line:
                last_import_line = node.end_lineno
                
    insertion_line = max(last_import_line, docstring_end_line)
    
    # If there is nothing, insert at 0
    lines = new_content_str.split('\n')
    lines.insert(insertion_line, '\n' + header_block)
    
    final_content = '\n'.join(lines)
    # Strip any consecutive extra newlines created by this process (optional but clean)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print(f"Updated {filepath}")

for root, dirs, files in os.walk('/home/hthakur/MultilingualLatentMAS'):
    # skip .git, .pytest_cache, etc.
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for file in files:
        if file.endswith('.py') and file != 'update_headers.py':
            process_file(os.path.join(root, file))
