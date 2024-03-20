import pandas as pd
from tsad.utils.evaluating.evaluating import evaluating
from typing import Any, Tuple


class SummaryReport:
    """ Idea is to create fast report for CPs task.
    """
    def __init__(self,
                 metric: str = "average_time",
                 window_width: str = "30 seconds",
                 portion: float = 0.1,
                 verbose: bool = False):
        self.metric = metric
        self.window_width = window_width
        self.portion = portion
        self.verbose = verbose

    def _evaluating(self, true: pd.Series, prediction: pd.Series) -> Tuple[Any]:
        """ Calculate CPD results based on TSAD library.

        Args:
            true: pd.Series of original data [0 or 1]
            prediction: pd.Series of preds data [0 or 1]

        Returns:
            tuple of metrics value.
        """
        return evaluating(true=true,
                          prediction=prediction,
                          metric=self.metric,
                          window_width=self.window_width,
                          portion=self.portion,
                          verbose=self.verbose)

    def create_report(self, df: pd.DataFrame, column_name_preds: str, column_name_original: str) -> pd.DataFrame:
        """ Generate one row report.

        Args:
            df: dataframe with preds and original data where index is timestamp data.
            column_name_preds: name of column where expected predicted CPs values.
            column_name_original: name of column where expected original CPs values.

        Returns:
            summary dataframe
        """
        true = df[column_name_original]
        preds = df[column_name_preds]
        res = self._evaluating(true, preds)
        if self.metric == "average_time":
            columns_out = ["Average time (average delay)",
                           "Missing changepoints",
                           "FPs",
                           "Number of true change-points"]
            out = pd.DataFrame(columns=columns_out, data=[res])
        else:
            raise NotImplementedError("Not yet any other metrics implemented")
        return out

    def create_big_report(self,
                          list_of_df: list[pd.DataFrame],
                          column_name_preds: str,
                          column_name_original: str) -> pd.DataFrame:
        """ Create one big dataframe report for many dataframes.

        Notes:
            column_name_preds and column_name_original should be the same in all dataframes!

        Args:
            list_of_df: list of dataframe with preds and original, data index is timestamp data.
            column_name_preds: name of column where expected predicted CPs values.
            column_name_original: name of column where expected original CPs values.

        Returns:
            summary dataframe
        """
        out = []
        for df in list_of_df:
            out.append(self.create_report(df, column_name_preds, column_name_original))
        return pd.concat(out).reset_index(drop=True)

