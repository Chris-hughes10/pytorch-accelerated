from pytest import fixture

from pytorch_accelerated.tracking import InMemoryRunHistory


@fixture
def run_history() -> InMemoryRunHistory:
    return InMemoryRunHistory()


def test_can_increment_epoch(run_history: InMemoryRunHistory):
    initial_epoch_count = run_history.current_epoch

    run_history.increment_epoch()
    incremented_once = run_history.current_epoch
    run_history.increment_epoch()
    incremented_twice = run_history.current_epoch

    assert incremented_once == initial_epoch_count +1
    assert incremented_twice == initial_epoch_count +2

def test_can_update_metric():
    pass

def test_can_get_metric_names():
    pass

def test_can_get_latest_metric():
    pass

def test_can_reset():
    pass