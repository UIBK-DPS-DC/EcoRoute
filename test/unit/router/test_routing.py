from unittest.mock import patch

import pytest

from router.router.config import RoutingConfig
from router.router.database import AvailableModel, AvailableRouter
from router.router.routing import Routing


@pytest.mark.parametrize(
    "models,routers",
    [
        (
            [
                AvailableModel(
                    id="12345",
                    mean_reward=5.0,
                    mean_response_time=5.0,
                    mean_energy=5.0,
                    pending_requests=5,
                )
            ],
            [],
        ),
        (
            [
                AvailableModel(
                    id="12345",
                    mean_reward=5.0,
                    mean_response_time=5.0,
                    mean_energy=5.0,
                    pending_requests=5,
                ),
                AvailableModel(
                    id="23456",
                    mean_reward=10.0,
                    mean_response_time=1.0,
                    mean_energy=2.0,
                    pending_requests=0,
                ),
            ],
            [],
        ),
        (
            [
                AvailableModel(
                    id="12345",
                    mean_reward=5.0,
                    mean_response_time=5.0,
                    mean_energy=5.0,
                    pending_requests=5,
                ),
                AvailableModel(
                    id="23456",
                    mean_reward=10.0,
                    mean_response_time=1.0,
                    mean_energy=2.0,
                    pending_requests=0,
                ),
            ],
            [
                AvailableRouter(
                    id="98765",
                    mean_reward=15.0,
                    mean_response_time=None,
                    mean_energy=5.0,
                    pending_requests=2,
                )
            ],
        ),
    ],
)
def test_predict(models, routers):
    routing = Routing(RoutingConfig())

    prediction = routing.predict(
        router_library=routers, llm_library=models, context={"task": "classification"}
    )

    assert len(prediction) == len(routers) + len(models)
    assert sum(prediction) == 1.0


@pytest.mark.parametrize(
    "models,routers",
    [
        (
            [
                AvailableModel(
                    id="12345",
                    mean_reward=5.0,
                    mean_response_time=5.0,
                    mean_energy=5.0,
                    pending_requests=5,
                )
            ],
            [],
        ),
        (
            [
                AvailableModel(
                    id="12345",
                    mean_reward=5.0,
                    mean_response_time=5.0,
                    mean_energy=5.0,
                    pending_requests=5,
                ),
                AvailableModel(
                    id="23456",
                    mean_reward=10.0,
                    mean_response_time=1.0,
                    mean_energy=2.0,
                    pending_requests=0,
                ),
            ],
            [],
        ),
        (
            [
                AvailableModel(
                    id="12345",
                    mean_reward=5.0,
                    mean_response_time=5.0,
                    mean_energy=5.0,
                    pending_requests=5,
                ),
                AvailableModel(
                    id="23456",
                    mean_reward=10.0,
                    mean_response_time=1.0,
                    mean_energy=2.0,
                    pending_requests=0,
                ),
            ],
            [
                AvailableRouter(
                    id="98765",
                    mean_reward=15.0,
                    mean_response_time=None,
                    mean_energy=5.0,
                    pending_requests=2,
                )
            ],
        ),
    ],
)
def test_learn(models, routers):

    routing = Routing(RoutingConfig())

    with patch.object(routing.vw, "learn") as mock_learn:
        routing.learn(
            router_library=routers,
            llm_library=models,
            context={"task": "classification"},
            selected_llm="12345",
            selection_prob=0.75,
            cost=12.0,
        )
        mock_learn.assert_called_once()


@pytest.mark.parametrize(
    "models,routers,training,expected",
    [
        (
            [
                AvailableModel(
                    id="12345",
                    mean_reward=5.0,
                    mean_response_time=5.0,
                    mean_energy=5.0,
                    pending_requests=5,
                )
            ],
            [],
            None,
            "shared | task:classification\n| model=12345 mean_reward=5.0 mean_response_time=5.0 mean_energy=5.0 pending_requests=5\n",
        ),
        (
            [
                AvailableModel(
                    id="12345",
                    mean_reward=5.0,
                    mean_response_time=5.0,
                    mean_energy=5.0,
                    pending_requests=5,
                ),
                AvailableModel(
                    id="23456",
                    mean_reward=10.0,
                    mean_response_time=1.0,
                    mean_energy=2.0,
                    pending_requests=0,
                ),
            ],
            [],
            None,
            "shared | task:classification\n| model=12345 mean_reward=5.0 mean_response_time=5.0 mean_energy=5.0 pending_requests=5\n| model=23456 mean_reward=10.0 mean_response_time=1.0 mean_energy=2.0 pending_requests=0\n",
        ),
        (
            [
                AvailableModel(
                    id="12345",
                    mean_reward=5.0,
                    mean_response_time=5.0,
                    mean_energy=5.0,
                    pending_requests=5,
                ),
                AvailableModel(
                    id="23456",
                    mean_reward=10.0,
                    mean_response_time=1.0,
                    mean_energy=2.0,
                    pending_requests=0,
                ),
            ],
            [
                AvailableRouter(
                    id="98765",
                    mean_reward=15.0,
                    mean_response_time=None,
                    mean_energy=5.0,
                    pending_requests=2,
                )
            ],
            {"23456": {"cost": 12.0, "prob": 0.75}},
            "shared | task:classification\n| model=12345 mean_reward=5.0 mean_response_time=5.0 mean_energy=5.0 pending_requests=5\n1:12.0:0.75 | model=23456 mean_reward=10.0 mean_response_time=1.0 mean_energy=2.0 pending_requests=0\n| router=98765  pending_requests=2",
        ),
    ],
)
def test_create_vw_format(models, routers, training, expected):
    routing = Routing(RoutingConfig())

    vw_string = routing._create_vw_format(
        router_library=routers,
        llm_library=models,
        context={"task": "classification"},
        training_data=training,
    )

    assert vw_string.strip() == expected.strip()
