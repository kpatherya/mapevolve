import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from mapevolve import cli, eval_routing
from test_tradeoff_policy import _payload


class EvalAndCliV2Tests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.profile = Path(self.dir.name, "p.json")
        self.profile.write_text(json.dumps(_payload()), encoding="utf-8")
        self.images = []
        for n in ("a.jpg", "b.jpg"):
            path = Path(self.dir.name, n)
            path.write_bytes(b"\xff\xd8x")
            self.images.append(str(path))

    def _run(self, fn, argv):
        out = StringIO()
        with redirect_stdout(out):
            code = fn(argv)
        return code, out.getvalue()

    def test_eval_report_savings_and_regret(self):
        code, out = self._run(eval_routing.main, ["--profile", str(self.profile)])
        self.assertEqual(code, 0)
        sweep = json.loads(out)["routing"]["sweep"]
        self.assertEqual(sweep[0]["model_id"], "frontier")
        self.assertEqual(sweep[0]["savings_vs_always_best"], 0.0)
        self.assertGreater(sweep[-1]["savings_vs_always_best"], 0.5)
        self.assertGreater(sweep[-1]["regret"], 0.0)

    def test_cli_v2_dry_run_records_policy_details(self):
        args = ["--request-id", "r", "--query", "Are these the same building?", "--dry-run",
                "--profiles-path", str(self.profile), "--policy", "utility", "--lam", "5"]
        for img in self.images:
            args += ["--image", img]
        code, out = self._run(cli.main, args)
        self.assertEqual(code, 0)
        trace = json.loads(out)
        self.assertEqual(trace["decision"]["backend_id"], "cheap")
        self.assertEqual(trace["policy"]["details"]["tradeoff"]["mode"], "utility")
        self.assertEqual(trace["classification"]["source"], "rules")

    def test_cli_v2_policy_rejects_v1_profile(self):
        code, _ = self._run(cli.main, ["--request-id", "r", "--query", "q", "--policy", "utility"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
