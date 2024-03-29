import numpy as np
from scipy.linalg import hankel
from models.SubspaceBased import SingularSequenceTransformer


class TestSingularSequenceTransformer:
    model = SingularSequenceTransformer()
    test_sequence = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    test_sequence_2 = np.array([111, 202, 903, 104, 205, 306, 707, 8, 9, 110])
    test_sequence_3 = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    history_matrix = hankel(test_sequence)
    future_matrix = hankel(test_sequence_2)
    future_matrix_2 = hankel(test_sequence_3)
    n_components = 1

    def test_get_hankel_matrix(self):
        """ Apply Hankel method over 1D time-series subsequence to transform it into matrix view.

        Return:
            Hankel matrix.
        """
        hankel_matrix = self.model.get_hankel_matrix(sequence=self.test_sequence)
        assert hankel_matrix.shape == (10, 10), "Matrix shape are equal to expected for test sequence."

    def test_sst_svd(self):
        """Apply singular sequence transformation algorithm with SVD.

        Return:
            distance between compared matrices.
        """
        distance_val_large: float = self.model._sst_svd(x_test=self.future_matrix,
                                                        x_history=self.history_matrix,
                                                        n_components=self.n_components)
        distance_val_small: float = self.model._sst_svd(x_test=self.future_matrix_2,
                                                        x_history=self.history_matrix,
                                                        n_components=self.n_components)
        assert distance_val_large > 0.05, "Large distance between two matrix in subspace."
        assert distance_val_small < 1e-2, "Small distance between two identical matrix in subspace."



