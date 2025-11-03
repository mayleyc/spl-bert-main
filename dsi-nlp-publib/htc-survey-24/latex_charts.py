import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np

import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np

class LatexTable:
    """
    Base LaTeX table class with booktabs formatting.
    """
    def __init__(self):
        self.rows = []
        self.header = []
        self.caption = None
        self.label = None

    def set_header(self, header):
        self.header = header

    def add_row(self, row):
        self.rows.append(row)

    def set_caption(self, caption):
        self.caption = caption

    def set_label(self, label):
        self.label = label

    def to_latex(self):
        if not self.header:
            raise ValueError("Header must be set before generating LaTeX table.")
        
        num_columns = len(self.header)
        col_spec = "l" + "c" * (num_columns - 1)
        
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\small")
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append("\\toprule")
        latex.append(" & ".join(self.header) + " \\\\")
        latex.append("\\midrule")
        
        for row in self.rows:
            if len(row) != num_columns:
                raise ValueError(f"Row '{row}' has a different number of columns than the header.")
            latex.append(" & ".join(map(str, row)) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        
        if self.caption:
            latex.append(f"\\caption{{{self.caption}}}")
        if self.label:
            latex.append(f"\\label{{{self.label}}}")
        
        latex.append("\\end{table}")
        return "\n".join(latex)
    
    def to_latex_split(self, max_metrics_per_table=3):
        """
        Split wide table into multiple vertical slices, each with at most `max_metrics_per_table` metrics.
        The 'Model' column is repeated in every slice.
        """
        if not self.header:
            raise ValueError("Header must be set before generating LaTeX table.")

        # exclude 'Model' from counting
        metrics = self.header[1:]
        slices = [metrics[i:i + max_metrics_per_table] 
                  for i in range(0, len(metrics), max_metrics_per_table)]

        tables = []
        for idx, metric_slice in enumerate(slices, start=1):
            # Include the 'Model' column in every slice
            slice_header = ['Model'] + metric_slice
            slice_rows = [[row[0]] + [row[self.header.index(col)] for col in metric_slice] 
                          for row in self.rows]
            
            # generate LaTeX for this slice
            tables.append(self._to_latex_partial(slice_header, slice_rows, part=idx))
        return "\n\n".join(tables)

    def _to_latex_partial(self, header, rows, part=1):
        num_columns = len(header)
        col_spec = "l" + "c" * (num_columns - 1)

        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\small")
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append("\\toprule")
        latex.append(" & ".join(header) + " \\\\")
        latex.append("\\midrule")

        for row in rows:
            latex.append(" & ".join(map(str, row)) + " \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")

        if self.caption:
            latex.append(f"\\caption{{{self.caption} (Part {part})}}")
        if self.label:
            latex.append(f"\\label{{{self.label}_part{part}}}")

        latex.append("\\end{table}")
        return "\n".join(latex)
    
    def latex_escape(self, s):
        """Escape LaTeX special characters in strings."""
        if not isinstance(s, str):
            return s
        return (s.replace('\\', r'\\')
                .replace('_', r'\_')
                .replace('%', r'\%')
                .replace('&', r'\&')
                .replace('#', r'\#')
                .replace('{', r'\{')
                .replace('}', r'\}')
                .replace('$', r'\$'))


class HTCTable(LatexTable):
    """
    Generates LaTeX table for HTC metrics from pickle files.
    """
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.data = defaultdict(dict)

    def process_pkls(self):
        pkls = [p for p in self.folder_path.iterdir() if p.suffix == ".pkl" and "htc" in p.stem]
        for fp in pkls:
            with open(fp, "rb") as f:
                pkl_data = pickle.load(f)

            model_name = self.latex_escape(fp.stem)
            for metric, values in pkl_data.items():
                metric = self.latex_escape(metric)
                mean, std = values
                self.data[metric][model_name] = f"{mean:.4f} $\\pm$ {std:.4f}"

    def generate_table(self):
        if not self.data:
            self.process_pkls()
            
        metrics = sorted(self.data.keys())
        models = sorted({model for metric_data in self.data.values() for model in metric_data.keys()})
        
        self.set_header(["Model"] + metrics)
        
        for model in models:
            row = [model]
            for metric in metrics:
                row.append(self.data[metric].get(model, "N/A"))
            self.add_row(row)

        self.set_caption("HTC performance metrics across models.")
        self.set_label("tab:htc_metrics")


class VioTable(LatexTable):
    """
    Generates LaTeX table for Violation count metrics from pickle files.
    """
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.data = defaultdict(lambda: defaultdict(dict))

    def process_pkls(self):
        pkls = [p for p in self.folder_path.iterdir() if p.suffix == ".pkl" and "vio" in p.stem]
        for fp in pkls:
            with open(fp, "rb") as f:
                pkl_data = pickle.load(f)

            model_name = self.latex_escape(fp.stem)
            for level, metrics in pkl_data.items():
                for metric, values in metrics.items():
                    metric = self.latex_escape(metric)
                    mean, std = values
                    #if not (np.isnan(mean) or np.isnan(std)):
                    self.data[model_name][level][metric] = f"{mean:.4f} $\\pm$ {std:.4f}"

    def generate_table(self):
        if not self.data:
            self.process_pkls()

        models = sorted(self.data.keys())
        levels = sorted({level for model_data in self.data.values() for level in model_data.keys()})
        metrics = sorted({metric for model_data in self.data.values() 
                                   for level_data in model_data.values() 
                                   for metric in level_data.keys()})

        header = ["Model"] + [f"{level}_{metric}" for level in levels for metric in metrics]
        self.set_header(header)

        for model in models:
            row = [model]
            for level in levels:
                for metric in metrics:
                    row.append(self.data[model].get(level, {}).get(metric, "N/A"))
            self.add_row(row)

        self.set_caption("Violation count metrics across models and difficulty levels.")
        self.set_label("tab:vio_metrics")


if __name__ == "__main__":
    # HTC Table Generation
    htc_table = HTCTable("dumps")
    htc_table.generate_table()
    latex_htc_code = htc_table.to_latex_split()

    with open("htc_metrics_table.tex", "w") as f:
        f.write(latex_htc_code)
    print("HTC LaTeX table exported to htc_metrics_table.tex")

    # Vio Table Generation
    vio_table = VioTable("dumps")
    vio_table.generate_table()
    latex_vio_code = vio_table.to_latex_split()

    with open("vio_metrics_table.tex", "w") as f:
        f.write(latex_vio_code)
    print("Vio LaTeX table exported to vio_metrics_table.tex")
