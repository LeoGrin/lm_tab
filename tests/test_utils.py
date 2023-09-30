import unittest
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pytest
from src.utils import FeaturesExtractor, FixedSizeSplit  # Replace `your_module` with the actual module name where FeaturesExtractor is defined

class TestFeaturesExtractor(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    def test_first_n_features(self):
        extractor = FeaturesExtractor(n_features=2, method="first")
        result = extractor.transform(self.X)
        expected = np.array([[1, 2], [5, 6], [9, 10]])
        np.testing.assert_array_equal(result, expected)

    def test_last_n_features(self):
        extractor = FeaturesExtractor(n_features=2, method="last")
        result = extractor.transform(self.X)
        expected = np.array([[3, 4], [7, 8], [11, 12]])
        np.testing.assert_array_equal(result, expected)

    def test_middle_n_features(self):
        extractor = FeaturesExtractor(n_features=2, method="middle")
        result = extractor.transform(self.X)
        expected = np.array([[2, 3], [6, 7], [10, 11]])
        np.testing.assert_array_equal(result, expected)

    def test_random_n_features(self):
        extractor = FeaturesExtractor(n_features=2, method="random")
        result = extractor.transform(self.X)
        self.assertEqual(result.shape, (self.X.shape[0], 2))

    def test_biggest_variance_n_features(self):
        extractor = FeaturesExtractor(n_features=2, method="biggest_variance")
        result = extractor.transform(self.X)
        expected = np.array([[3, 4], [7, 8], [11, 12]])  # In this specific case, the last two features have the highest variance
        np.testing.assert_array_equal(result, expected)



def test_fixed_size_split():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    fss = FixedSizeSplit(n_train=3, n_splits=2, random_state=42)
    
    for train, test in fss.split(X, y):
        assert len(train) == 3
        assert len(test) == 2
    
    fss = FixedSizeSplit(n_train=2, n_test=1, n_splits=2, random_state=42)
    for train, test in fss.split(X, y):
        assert len(train) == 2
        assert len(test) == 1

def test_n_train_too_large():
    with pytest.raises(ValueError):
        for train, test in FixedSizeSplit(n_train=6, n_splits=2).split(np.array([[1], [2], [3], [4], [5]])):
            pass

def test_n_test_too_large():
    with pytest.raises(ValueError):
        for train, test in FixedSizeSplit(n_train=4, n_test=2, n_splits=2).split(np.array([[1], [2], [3], [4], [5]])):
            pass

def test_cross_val_score_n_jobs():
    # create a rng
    rng = np.random.default_rng(42)
    y = rng.random(100)
    X = rng.random((100, 10))
    fss = FixedSizeSplit(n_train=80, n_splits=5, random_state=42)
    
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=fss, n_jobs=5)  # n_jobs > 1

    assert len(np.unique(scores)) == len(scores)  # All scores should be unique

if __name__ == "__main__":
    unittest.main()