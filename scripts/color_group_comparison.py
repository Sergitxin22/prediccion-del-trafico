import argparse
import csv
import html
import math
from pathlib import Path

from compare_group_graph import CASES, FIELDS, SETTINGS


GROUPS = tuple(label for label, _, _ in CASES if label not in ("Ba1", "Ba2"))
COLORS = {
    "rojo": "No mejora ninguna base",
    "amarillo": "Mejora solo Ba1",
    "naranja": "Mejora solo Ba2",
    "verde": "Mejora ambas bases",
    "gris": "Igual a ambas bases",
}


def classify(value, ba1, ba2):
    better1, better2 = value < ba1, value < ba2
    if better1 and better2:
        return "verde"
    if better1:
        return "amarillo"
    if better2:
        return "naranja"
    if value == ba1 == ba2:
        return "gris"
    return "rojo"


def label(value, baseline):
    return "si" if value < baseline else "igual" if value == baseline else "no"


def decorate(source):
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("El CSV no contiene las diez variantes y sus cinco parametros esperados.")
        rows = list(reader)
    if not rows:
        raise ValueError("El CSV de comparacion esta vacio.")
    decorated, seen = [], set()
    for row in rows:
        key = tuple(row[field] for field in SETTINGS)
        if key in seen:
            raise ValueError("Hay configuraciones duplicadas.")
        seen.add(key)
        scores = {field: float(row[field]) for field in FIELDS if field not in SETTINGS}
        if any(not math.isfinite(value) or value < 0 for value in scores.values()):
            raise ValueError("Las puntuaciones deben ser finitas y no negativas.")
        result = dict(row)
        total = base1 = base2 = 0
        for group in GROUPS:
            value = scores[group]
            result[f"{group}_mejora_Ba1"] = label(value, scores["Ba1"])
            result[f"{group}_mejora_Ba2"] = label(value, scores["Ba2"])
            result[f"{group}_color"] = classify(value, scores["Ba1"], scores["Ba2"])
            base1 += value < scores["Ba1"]
            base2 += value < scores["Ba2"]
            total += value < scores["Ba1"] or value < scores["Ba2"]
        result.update(mejoras_total=total, mejoras_Ba1=base1, mejoras_Ba2=base2)
        decorated.append(result)
    return decorated


def render_table(rows):
    header = "".join(f"<th scope='col'>{html.escape(field)}</th>" for field in FIELDS)
    body = []
    for row in sorted(rows, key=lambda item: -item["mejoras_total"]):
        cells = [f"<td>{html.escape(row[field])}</td>" for field in SETTINGS]
        for field in ("Ba1", "Ba2"):
            cells.append(f"<td class='base'>{float(row[field]):.4f}</td>")
        for group in GROUPS:
            color = row[f"{group}_color"]
            caption = COLORS[color]
            cells.append(
                f"<td class='{color}' title='{caption}; MSE relativo: {float(row[group]):.8f}'>"
                f"<strong>{float(row[group]):.4f}</strong><small>{caption}</small></td>"
            )
        cells.append(f"<td class='total'><strong>{row['mejoras_total']}/8</strong></td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    possible = 2 * len(rows)
    totals = {
        group: sum(row[f"{group}_mejora_Ba1"] == "si" for row in rows)
        + sum(row[f"{group}_mejora_Ba2"] == "si" for row in rows)
        for group in GROUPS
    }
    footer = (
        f"<th scope='row' colspan='{len(SETTINGS)}'>Victorias por columna (Ba1 + Ba2)</th>"
        "<td>&mdash;</td><td>&mdash;</td>"
        + "".join(f"<td class='column-total'><strong>{totals[group]}/{possible}</strong></td>" for group in GROUPS)
        + f"<td class='column-total'><strong>{sum(totals.values())}/{possible * len(GROUPS)}</strong></td>"
    )
    return (f"<div class='table-wrap'><table><thead><tr>{header}<th scope='col'>Mejoras</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody><tfoot><tr>{footer}</tr></tfoot></table></div>")


def render_html(rows, horizons=None):
    legends = "".join(f"<span class='{color}'>{html.escape(caption)}</span>" for color, caption in COLORS.items())
    possible = 2 * len(rows)
    if horizons is None:
        tables = render_table(rows)
    else:
        if tuple(horizons) != (1, 2, 3, 4) or any(not values or len(values) != len(rows) for values in horizons.values()):
            raise ValueError("Expected four nonempty horizons with the same number of configurations")
        tables = (
            "<nav class='horizon-nav' aria-label='Horizontes'>"
            + "".join(f"<a href='#h{step}'>+{step * 15} minutos</a>" for step in horizons)
            + "</nav>"
            + "".join(
                f"<section class='horizon' id='h{step}'><h2>+{step * 15} minutos</h2>"
                f"<p>Resultados exclusivamente a +{step * 15} minutos; "
                "MSE relativo a la persistencia de este mismo horizonte.</p>"
                + render_table(values) + "</section>"
                for step, values in horizons.items()
            )
        )
    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agrupaciones frente a sensores individuales</title>
<style>
:root {{ color-scheme: light; --ink:#172832; --muted:#526771; --border:#dce5e6; --page:#f3f6f5; font-family: Georgia, serif; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); }}
main {{ max-width:1600px; margin:auto; padding:36px 24px 64px; }}
header {{ border-top:6px solid #106953; padding-top:25px; }}
h1 {{ font-size:clamp(30px,4vw,52px); letter-spacing:-.04em; margin:0 0 14px; }}
p {{ max-width:930px; line-height:1.55; color:var(--muted); }}
.legend {{ display:flex; flex-wrap:wrap; gap:9px; margin:24px 0; }}
.legend span {{ padding:8px 12px; border-radius:8px; font:600 13px Georgia,serif; }}
.horizon-nav {{ display:flex; flex-wrap:wrap; gap:9px; margin:22px 0 28px; }}
.horizon-nav a {{ padding:9px 14px; border:1px solid #9ebfb3; border-radius:8px; color:#194e43; background:white; text-decoration:none; font-weight:bold; }}
.horizon-nav a:hover,.horizon-nav a:focus {{ background:#e7f3ee; }}
.horizon {{ margin:0 0 48px; scroll-margin-top:18px; }}
.horizon h2 {{ font-size:26px; margin:0 0 8px; }}
.table-wrap {{ overflow:auto; border:1px solid var(--border); border-radius:12px; max-height:72vh; background:white; }}
table {{ border-collapse:separate; border-spacing:0; width:100%; min-width:1480px; font:13px Georgia,serif; }}
th,td {{ padding:11px 9px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap; }}
th {{ position:sticky; top:0; z-index:2; background:#194e43; color:#fff; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
td:nth-child(-n+5) {{ background:#f5f8f7; }}
td small {{ display:block; font-size:10px; margin-top:3px; }}
.base {{ background:#e9eff0; font-weight:bold; }}
.rojo {{ background:#fbe2de; color:#822f29; }}
.amarillo {{ background:#fff1b8; color:#664c00; }}
.naranja {{ background:#ffe2bb; color:#84420a; }}
.verde {{ background:#d5efdf; color:#17543a; }}
.gris {{ background:#e8edef; color:#4b5d64; }}
.total {{ background:#e7f3ee; color:#13533f; font-size:16px; }}
tfoot th,tfoot td {{ position:sticky; bottom:0; z-index:2; background:#194e43; color:white; border-top:2px solid #8cb9a6; border-bottom:0; }}
tfoot td:nth-child(-n+5) {{ background:#194e43; }}
tfoot th {{ text-align:left; }}
.column-total {{ font-size:15px; font-variant-numeric:tabular-nums; }}
tr:hover td {{ outline:1px solid #8faeaa; outline-offset:-1px; }}
</style>
</head>
<body><main><header><h1>¿La agrupaci&oacute;n mejora al sensor?</h1>
<p>Una fila conserva arquitectura, modelo y par&aacute;metros. Cada celda compara su MSE
con <b>Ba1 y Ba2 de esa misma fila</b>. Menor MSE relativo es mejor.
Las ocho celdas c se punt&uacute;an una vez cada una si mejoran a al menos una base;
verde cuenta una vez, no dos. <b>Al final de cada columna y de cada horizonte</b> se suman ambas
comparaciones de todas las filas: el verde aporta 2, el amarillo o naranja 1
y el rojo 0. Por eso el denominador es 2 &times; {len(rows)} = {possible}.
Los empates no se consideran mejora.</p></header>
<div class="legend" aria-label="Significado de los colores">{legends}</div>
<p>Ba1 = sensores individuales / cercan&iacute;a; Ba2 = sensores individuales / correlaci&oacute;n.
c1 = proximidad 50, c2 = proximidad 500, c3 = proximidad + comportamiento 500,
c4 = proximidad + comportamiento 50; a1 = cercan&iacute;a, a2 = correlaci&oacute;n.
Son errores en los mismos sensores individuales y fechas de prueba, no sobre el agregado.</p>
{tables}
</main></body></html>
"""


def generate(source, output):
    rows = decorate(source)
    columns = [*FIELDS, *(field for group in GROUPS for field in
                           (f"{group}_mejora_Ba1", f"{group}_mejora_Ba2", f"{group}_color")),
               "mejoras_total", "mejoras_Ba1", "mejoras_Ba2"]
    page = render_html(rows)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparacion_coloreada.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparacion_coloreada.html").write_text(page, encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Clasifica las comparaciones frente a ambas bases sin recalcular modelos.")
    parser.add_argument("--source", required=True, type=Path, help="paired_mse_ratio.csv")
    parser.add_argument("--output", required=True, type=Path, help="Carpeta nueva para CSV y HTML")
    args = parser.parse_args()
    rows = generate(args.source, args.output)
    print(f"{len(rows)} configuraciones clasificadas: {args.output}")


if __name__ == "__main__":
    main()
