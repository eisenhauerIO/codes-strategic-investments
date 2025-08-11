import papermill as pm

notebooks_to_run = [
    "Tutorial.ipynb",
    "Visualization.ipynb"
]

for nb in notebooks_to_run:
    print(f"Running: {nb}")
    pm.execute_notebook(
        input_path=nb,
        output_path=nb.replace(".ipynb", "-executed.ipynb"),
    )