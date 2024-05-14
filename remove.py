import nbformat

def delete_markdown_cells(notebook_path):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Iterate over each cell
    cells_to_keep = []
    for cell in nb['cells']:
        # Only keep non-Markdown cells
        if cell['cell_type'] != 'markdown':
            cells_to_keep.append(cell)
    
    # Replace the notebook's cells with the filtered ones
    nb['cells'] = cells_to_keep
    
    # Save the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

# Specify the path to your Jupyter Notebook file
notebook_path = 'notebook.ipynb'

# Call the function to delete Markdown cells
delete_markdown_cells(notebook_path)
