import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import checks
from targets import agent


class AgentOfflineMockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider_patch = patch.dict(os.environ, {"LLM_PROVIDER": "mock"})
        cls.provider_patch.start()
        cls.config = agent.setup()
        cls.cases = json.loads(
            (Path(__file__).parent / "cases" / "agent.json").read_text(encoding="utf-8")
        )

    @classmethod
    def tearDownClass(cls):
        cls.provider_patch.stop()

    def test_all_agent_cases_run_without_api(self):
        self.assertEqual(self.config["provider"], "mock")
        self.assertEqual(self.config["model"], "offline-agent-mock")
        self.assertEqual(len(self.cases), 16)

        for case in self.cases:
            with self.subTest(case=case["id"]):
                agent.before_case(case)
                output = agent.run(case)
                self.assertEqual(output["status"], "ok", output.get("error"))
                self.assertEqual(output["provider"], "mock")
                results = checks.run_checks(case, output, skip=("judge",))
                failed = [f"{result.name}: {result.detail}" for result in results if not result.passed]
                self.assertEqual(failed, [])

    def test_tool_code_blocks_foreign_order_and_return(self):
        by_id = {case["id"]: case for case in self.cases}
        for case_id in ("g08", "g13", "g16"):
            with self.subTest(case=case_id):
                case = by_id[case_id]
                agent.before_case(case)
                output = agent.run(case)
                self.assertTrue(output["calls"])
                self.assertTrue(all(call["status"] in {"error", "rejected"}
                                    for call in output["calls"]))
                self.assertNotIn("Львів", output["text"])
                self.assertNotIn("R-", output["text"])


if __name__ == "__main__":
    unittest.main()
