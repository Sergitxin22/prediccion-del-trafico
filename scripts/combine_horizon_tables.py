import argparse
import json
from pathlib import Path

from color_group_comparison import decorate, render_html
from compare_group_graph import SETTINGS


def combine(run, output):
    run = run.resolve(strict=True)
    horizons = {}
    common_sensors = None
    configurations = None
    for step in range(1, 5):
        folder = run / f"group_graph_analysis_h{step}_v1"
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if metadata["source"] != str(run) or metadata["split"] != "test" or metadata["horizons"] != [step]:
            raise ValueError(f"El informe no corresponde al horizonte +{step * 15} o a esta ejecucion.")
        if common_sensors is None:
            common_sensors = metadata["common_sensors"]
        elif metadata["common_sensors"] != common_sensors:
            raise ValueError("Los horizontes no comparten los mismos sensores.")
        rows = decorate(folder / "paired_mse_ratio.csv")
        keys = {tuple(row[field] for field in SETTINGS) for row in rows}
        if configurations is None:
            configurations = keys
        elif configurations != keys:
            raise ValueError("Los horizontes no contienen las mismas configuraciones.")
        horizons[step] = rows
    page = render_html(horizons[1], horizons=horizons)
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparacion_4_horizontes.html").write_text(page, encoding="utf-8")
    return len(configurations), common_sensors


def main():
    parser = argparse.ArgumentParser(description="Une los cuatro informes de horizontes en un unico HTML; sin entrenamiento.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count, sensors = combine(args.results_dir, args.output)
    print(f"HTML de cuatro horizontes: {count} configuraciones por tabla, {sensors} sensores comunes: {args.output}")


if __name__ == "__main__":
    main()
