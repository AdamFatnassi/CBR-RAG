import unittest
import pandas as pd
from src.augmentation import build_prompt, build_prompt_from_context
from src.rag_generator import CBR_RAG_Generator


class TestRAG(unittest.TestCase):
    def setUp(self):
        self.sample_table = pd.DataFrame([{
            "rank": 1, "similarity": 0.931, "outcome": "no", "reliability": 1.0,
            "age": 38.0, "job": "admin.", "marital": "married", "education": "university.degree",
            "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
            "month": "may", "day_of_week": "thu", "campaign": 2.0, "previous": 0.0,
            "poutcome": "nonexistent", "pdays_group": "none", "emp.var.rate": 1.1,
            "cons.price.idx": 93.749, "cons.conf.idx": -41.8, "euribor3m": 4.857,
            "nr.employed": 5191.0, "same_categories": 9
        }])

        self.context = {
            "query_data": {
                "age": 38, "job": "admin.", "marital": "married", "education": "university.degree",
                "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
                "month": "may", "day_of_week": "thu", "campaign": 2, "previous": 0,
                "poutcome": "nonexistent", "pdays_group": "none", "emp.var.rate": 1.1,
                "cons.price.idx": 93.749, "cons.conf.idx": -41.8, "euribor3m": 4.857, "nr.employed": 5191.0
            },
            "prediction": 0,
            "score": 0.03316,
            "retrieved_cases": self.sample_table
        }

    def test_build_prompt_with_context(self):
        sys_p, usr_p, decision, truth = build_prompt(self.context)
        self.assertIn("You are a banking marketing analyst", sys_p)
        self.assertIn("New client: Client aged 38", usr_p)
        self.assertIn("Proposed decision: do not target", usr_p)
        self.assertEqual(decision, "no")

    def test_generator_simulation_with_context(self):
        gen = CBR_RAG_Generator(use_api=False)
        out = gen.generate_explanation(self.context)
        self.assertIn("MODE SIMULATION LLM", out)
        self.assertIn("Décision calculée : no", out)


if __name__ == "__main__":
    unittest.main()
