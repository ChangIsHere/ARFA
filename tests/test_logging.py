from src.common.logger import get_logger


def test_logger_returns_named_logger():
    assert get_logger("arfa.test").name == "arfa.test"
