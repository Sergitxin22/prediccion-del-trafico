import csv
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from combine_horizon_tables import combine
from compare_group_graph import FIELDS


class CombineHorizonTablesTests(unittest.TestCase):
    def test_four_tables_have_independent_footers_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            row = dict.fromkeys(FIELDS, "1")
            row.update(architecture="pfm", model="gcn_lstm", time_alpha="0.2", layers="2", hidden_feats="32")
            for step in range(1, 5):
                folder = run / f"group_graph_analysis_h{step}_v1"
                folder.mkdir()
                (folder / "metadata.json").write_text(json.dumps({
                    "source": str(run), "split": "test", "horizons": [step], "common_sensors": 2
                }))
                with (folder / "paired_mse_ratio.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerow(dict(row, c1a1="0.5" if step in (1, 4) else "2"))
            output = run / "combined"
            self.assertEqual(combine(run, output), (1, 2))
            page = (output / "comparacion_4_horizontes.html").read_text()
            self.assertEqual(page.count("<table>"), 4)
            self.assertEqual(page.count("<tfoot>"), 4)
            self.assertIn("id='h1'><h2>+15 minutos</h2>", page)
            self.assertIn("id='h4'><h2>+60 minutos</h2>", page)
            footers = re.findall(r"<tfoot>(.*?)</tfoot>", page, flags=re.S)
            first_column = [re.search(r"class='column-total'><strong>([0-9]+/[0-9]+)</strong>", footer).group(1)
                            for footer in footers]
            self.assertEqual(first_column, ["2/2", "0/2", "0/2", "2/2"])
            with self.assertRaises(FileExistsError):
                combine(run, output)

    def test_mismatched_horizon_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            folder = run / "group_graph_analysis_h1_v1"
            folder.mkdir()
            (folder / "metadata.json").write_text(json.dumps({
                "source": str(run), "split": "test", "horizons": [4], "common_sensors": 2
            }))
            output = run / "combined"
            with self.assertRaises(ValueError):
                combine(run, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
