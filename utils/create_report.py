import pandas as pd
from tsad.evaluating.evaluating import evaluating  # F1 score evaluation
from typing import Dict, List


class ChangePointDetectionReport:
    def __init__(self,
                 df_original: pd.DataFrame = None):
        self.average_result = None
        self.nab_result = None
        self.predicted_list = None
        self.original_list = None
        self.df_original = df_original

    def fit(self, predicted_list: List[int], original_list: List[int]) -> object:
        self.predicted_list = predicted_list
        self.original_list = original_list
        self.average_result = self.create_report(self.tsad_average())
        self.nab_result = self.nab_result(predicted_list, original_list)
        return self

    def tsad_average(self):
        average_time, missed_cp, fps, true_anomalies = self.calculate_average_tsad()
        tp = true_anomalies - missed_cp
        return {'Time_Delta': average_time, 'Missed_CP': missed_cp,
                'FPs': fps, 'True_Anomalies_Count': true_anomalies,
                'precision': self.calculate_precision(tp, fps),
                'recall': self.calculate_recall(tp, missed_cp), 'F1': self.calculate_f1(tp, fps, missed_cp)}

    def calculate_average_tsad(self) -> List[int, int, int, int]:
        """ Average TSAD calculation based on predicted and original list.

        Returns
         A list of params as average_time, missed cp, FPs, true anomalies.
        """
        return evaluating(self.original_list, self.predicted_list,
                          metric='average_time', numenta_time='30 sec', verbose=False)

    def calculate_f1(self, tp: int, fps:int, missed_cp: int, ) -> float:
        precision = self.calculate_precision(tp, fps)
        recall = self.calculate_recall(tp, missed_cp)
        return 2 * precision * recall / (precision + recall) if (precision != 0) or (recall != 0) else 0

    @staticmethod
    def calculate_recall(tp: int, missed_cp: int) -> float:
        return tp / (tp + missed_cp)

    @staticmethod
    def calculate_precision(tp: int, fps: int) -> float:
        return tp / (tp + fps)

    @staticmethod
    def create_report(experiment_results: Dict[str]) -> pd.DataFrame:
        return pd.DataFrame.from_dict(experiment_results, orient='index').fillna(0)

    @staticmethod
    def tsad_nab(predicted_list: List[int],
                 original_list: List[int]):
        return evaluating(original_list, predicted_list,
                          metric='nab', numenta_time='30 sec', verbose=False)
