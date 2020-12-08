import unittest
from linajea.evaluation import evaluate_divisions


class TestDivisionEval(unittest.TestCase):
    def test_division_eval(self):
        gt_divisions = {
                2: [[0, 0, 0, 1]],
                3: [[5, 5, 5, 2],  # TP
                    [10, 10, 10, 3]],  # FN -> TP
                4: [[1, 10, 100, 4]],
                }
        target_frame = 3
        matching_threshold = 1.5

        rec_divisions = {
                3: [[0, 0, 0, 10],  # FP -> TP
                    [5, 5, 5, 20],  # TP
                    [50, 50, 50, 50]],  # FP
                4: [[10, 10, 10, 30]],
                }

        reports = evaluate_divisions(
                gt_divisions,
                rec_divisions,
                target_frame,
                matching_threshold,
                frame_buffer=1)
        self.assertEqual(len(reports), 2)
        no_buffer = reports[0]
        # gt_total, rec_total, FP, FN, Prec, rec, F1
        self.assertEqual(no_buffer[0:6], (2, 3, 2, 1, 1/3, 1/2))
        one_buffer = reports[1]
        self.assertEqual(one_buffer[0:6], (2, 3, 1, 0, 2/3, 1))
