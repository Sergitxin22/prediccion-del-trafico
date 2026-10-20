import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from color_group_comparison import GROUPS, classify, generate, render_html
from compare_group_graph import FIELDS


class ColorGroupComparisonTests(unittest.TestCase):
    def test_color_rules_include_both_baselines_and_ties(self):
        self.assertEqual(classify(1.5, 2., 1.), "amarillo")
        self.assertEqual(classify(1.5, 1., 2.), "naranja")
        self.assertEqual(classify(.5, 1., 2.), "verde")
        self.assertEqual(classify(2., 1., 1.), "rojo")
        self.assertEqual(classify(1., 1., 1.), "gris")
        self.assertEqual(classify(1., 1., 2.), "naranja")
        self.assertEqual(classify(1., 2., 1.), "amarillo")

    def test_csv_and_html_count_each_positive_cell_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "paired_mse_ratio.csv"
            first = dict(zip(FIELDS, ("<script>", "gcn_lstm", "0", "1", "32",
                                      "1", "2", ".5", ".5", "2.5", "1", "2", "1", ".5", "2.5")))
            second = dict(first, architecture="pfm", Ba1="2", Ba2="1", c1a1="1.5", c1a2=".5",
                          c2a1="2.5", c2a2="2.5", c3a1="2.5", c3a2="2.5", c4a1="2.5", c4a2="1")
            equal = dict(first, architecture="other", Ba1="1", Ba2="1", **{key: "1" for key in GROUPS})
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows((first, second, equal))
            output = root / "new"
            rows = generate(source, output)
            self.assertEqual([row["mejoras_total"] for row in rows], [5, 3, 0])
            self.assertEqual((rows[0]["mejoras_Ba1"], rows[0]["mejoras_Ba2"]), (3, 5))
            self.assertEqual(rows[0]["c1a1_color"], "verde")
            self.assertEqual(rows[0]["c1a1_mejora_Ba1"], "si")
            self.assertEqual(rows[0]["c1a1_mejora_Ba2"], "si")
            self.assertEqual(rows[0]["c2a2_color"], "naranja")
            self.assertEqual(rows[1]["c1a1_color"], "amarillo")
            self.assertEqual(rows[2]["c1a1_color"], "gris")
            with (output / "comparacion_coloreada.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["mejoras_total"], "5")
            page = (output / "comparacion_coloreada.html").read_text(encoding="utf-8")
            self.assertIn("class='verde'", page)
            self.assertIn("class='rojo'", page)
            self.assertIn("&lt;script&gt;", page)
            self.assertNotIn("<script>", page)
            self.assertNotIn("summary-row", page)
            self.assertEqual(page.count("<tfoot>"), 1)
            self.assertIn("<strong>3/6</strong>", page)
            self.assertIn("<strong>3/4</strong>", render_html(rows[:2]))
            with self.assertRaises(FileExistsError):
                generate(source, output)

    def test_invalid_scores_are_rejected_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "paired_mse_ratio.csv"
            row = dict.fromkeys(FIELDS, "1")
            row["c1a1"] = "nan"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(row)
            output = root / "new"
            with self.assertRaises(ValueError):
                generate(source, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
