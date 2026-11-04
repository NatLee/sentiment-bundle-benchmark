import pickle

import pytest

from sentiment_bundle_benchmark.safe_pickle import ALLOWED_GLOBALS, UnsafePickleError, load_pickle


class _Exploit:
    def __init__(self, target):
        self.target = target

    def __reduce__(self):
        # Loading this would create a file if the unpickler let builtins.open through.
        return (open, (str(self.target), "w"))


def test_loads_plain_containers(tmp_path):
    path = tmp_path / "plain.pkl"
    data = (["a review"], [1], ["another"], [0])
    path.write_bytes(pickle.dumps(data, protocol=3))
    assert load_pickle(path) == data


def test_refuses_globals_outside_the_allow_list(tmp_path):
    marker = tmp_path / "should-not-exist"
    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(_Exploit(marker)))
    with pytest.raises(UnsafePickleError, match="open"):
        load_pickle(path)
    assert not marker.exists()


def test_allow_list_only_covers_dataframe_building_blocks():
    modules = {module.split(".")[0] for module, _ in ALLOWED_GLOBALS}
    assert modules <= {"builtins", "numpy", "pandas"}
    risky = {"eval", "exec", "open", "system", "getattr", "__import__", "compile"}
    assert not any(name in risky for _, name in ALLOWED_GLOBALS)
