import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tea_nir.preprocess import Preprocessor
from tea_nir.model import TeaScreeningModel


def test_snv_centres_rows():
    X = np.random.RandomState(0).rand(10, 256) + 1
    Z = Preprocessor("snv").fit_transform(X)
    assert np.allclose(Z.mean(1), 0, atol=1e-9)


def test_screen_flags_out_of_domain():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 256) * 0.1 + 0.3
    y = (X[:, 100] > np.median(X[:, 100])).astype(int)
    m = TeaScreeningModel(n_components=5).fit(X, y)
    far = np.full((1, 256), 5.0)
    assert m.screen(far)[0]["verdict"] == "OUT OF DOMAIN"
    assert len(m.screen(X)) == 40
