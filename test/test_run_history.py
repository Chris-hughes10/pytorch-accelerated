from pytest import fixture, raises

from pytorch_accelerated.tracking import InMemoryRunHistory


@fixture
def run_history() -> InMemoryRunHistory:
    return InMemoryRunHistory()


def test_can_increment_epoch(run_history: InMemoryRunHistory):
    initial_epoch_count = run_history.current_epoch

    run_history._increment_epoch()
    incremented_once = run_history.current_epoch
    run_history._increment_epoch()
    incremented_twice = run_history.current_epoch

    assert incremented_once == initial_epoch_count + 1
    assert incremented_twice == initial_epoch_count + 2


def test_can_get_metric_values(run_history: InMemoryRunHistory):
    metric_name = "my_metric"
    expected_metric_values = [0, 1, 2]
    run_history._metrics[metric_name].extend(expected_metric_values)

    metric_values = run_history.get_metric_values(metric_name)

    assert metric_values == expected_metric_values


def test_can_update_metric(run_history: InMemoryRunHistory):
    metric_name = "my_metric"
    expected_metric_values = [0, 1, 2]

    for value in expected_metric_values:
        run_history.update_metric(metric_name, value)

    assert run_history.get_metric_values(metric_name) == expected_metric_values


def test_can_get_latest_metric(run_history: InMemoryRunHistory):
    metric_name = "my_metric"
    metric_values = [0, 1, 2]
    for value in metric_values:
        run_history.update_metric(metric_name, value)

    latest_metric = run_history.get_latest_metric(metric_name)

    assert latest_metric == metric_values[-1]


def test_can_not_get_latest_metric_if_not_recorded(run_history: InMemoryRunHistory):
    metric_name = "my_metric"
    with raises(ValueError):
        run_history.get_latest_metric(metric_name)


def test_can_get_metric_names(run_history: InMemoryRunHistory):
    expected_metric_names = {"my_metric_1", "my_metric_2"}
    expected_metric_values = [0, 1]
    for name, value in zip(expected_metric_names, expected_metric_values):
        run_history.update_metric(name, value)

    metric_names = run_history.get_metric_names()

    assert metric_names == expected_metric_names


def test_can_reset(run_history: InMemoryRunHistory):
    metric_names = {"my_metric_1", "my_metric_2"}
    metric_values = [0, 1]
    for name, value in zip(metric_names, metric_values):
        run_history.update_metric(name, value)

    run_history.reset()

    assert len(run_history.get_metric_names()) == 0
