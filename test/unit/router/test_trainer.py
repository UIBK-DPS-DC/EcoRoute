from unittest.mock import patch

import pytest

from router.router.config import TrainerConfig
from router.router.database import TrainingSample
from router.router.trainer import Trainer
from datetime import datetime


@pytest.fixture(scope="module")
def trainer():
    with (
        patch("router.router.trainer.DuckDB") as mock_duckdb,
        patch("router.router.trainer.Routing") as mock_routing,
    ):
        mock_routing.return_value.learn.return_value = None

        mock_duckdb.return_value.metrics_since_last_training_batch.return_value = 1
        mock_duckdb.return_value.get_training_batch.return_value = []
        mock_duckdb.return_value.set_processed.return_value = None

        trainer = Trainer(
            duckdb=mock_duckdb.return_value,
            routing=mock_routing.return_value,
            config=TrainerConfig(),
        )

        return trainer


@pytest.mark.parametrize(
    "task,response,target,expected,dataset,subset",
    [
        ("classification", "0", "0", 1.0, "glue_mnli", None),
        ("classification", "1", "0", 0.0, "glue_mnli", None),
        ("classification", "non numeric answer", "0", 0.0, "glue_mnli", None),
        ("classification", "0", "0", 1.0, "glue_qqp", None),
        ("classification", "1", "0", 0.0, "glue_qqp", None),
        ("classification", "1", "1", 1.0, "glue_sst2", None),
        ("classification", "0", "1", 0.0, "glue_sst2", None),
        (
            "reasoning",
            "The answer to the question is 105.",
            "You save $5 per chair with the 25% off sale.\nOn sale chairs cost $15 each because 20 - 5 = <<20-5=15>>15\nYou bought 3 chairs above the 5 required before the extra savings starts because 8 - 5 = <<8-5=3>>3\nYou save an extra $5 on these chairs because 15 x (1/3) = <<15*(1/3)=5>>5\nThe first five chairs cost $75 in total because 5 x 15 = <<5*15=75>>75\nThe final three chairs cost $30 in total because 3 x 10 = <<3*10=30>>30\nIn total they cost $105 because 75 + 30 = <<75+30=105>>105\n#### 105",
            1.0,
            "gsm8k",
            None,
        ),
        (
            "reasoning",
            "The answer to the question is 89.",
            "You save $5 per chair with the 25% off sale.\nOn sale chairs cost $15 each because 20 - 5 = <<20-5=15>>15\nYou bought 3 chairs above the 5 required before the extra savings starts because 8 - 5 = <<8-5=3>>3\nYou save an extra $5 on these chairs because 15 x (1/3) = <<15*(1/3)=5>>5\nThe first five chairs cost $75 in total because 5 x 15 = <<5*15=75>>75\nThe final three chairs cost $30 in total because 3 x 10 = <<3*10=30>>30\nIn total they cost $105 because 75 + 30 = <<75+30=105>>105\n#### 105",
            0.0,
            "gsm8k",
            None,
        ),
        (
            "reasoning",
            "The answer to the question is 89.",
            "You save $5 per chair with the 25% off sale.\nOn sale chairs cost $15 each because 20 - 5 = <<20-5=15>>15\nYou bought 3 chairs above the 5 required before the extra savings starts because 8 - 5 = <<8-5=3>>3\nYou save an extra $5 on these chairs because 15 x (1/3) = <<15*(1/3)=5>>5\nThe first five chairs cost $75 in total because 5 x 15 = <<5*15=75>>75\nThe final three chairs cost $30 in total because 3 x 10 = <<3*10=30>>30\nIn total they cost $105 because 75 + 30 = <<75+30=105>>105",
            0.0,
            "gsm8k",
            None,
        ),
        (
            "reasoning",
            "This answer does not contain a number.",
            "You save $5 per chair with the 25% off sale.\nOn sale chairs cost $15 each because 20 - 5 = <<20-5=15>>15\nYou bought 3 chairs above the 5 required before the extra savings starts because 8 - 5 = <<8-5=3>>3\nYou save an extra $5 on these chairs because 15 x (1/3) = <<15*(1/3)=5>>5\nThe first five chairs cost $75 in total because 5 x 15 = <<5*15=75>>75\nThe final three chairs cost $30 in total because 3 x 10 = <<3*10=30>>30\nIn total they cost $105 because 75 + 30 = <<75+30=105>>105\n#### 105",
            0.0,
            "gsm8k",
            None,
        ),
        (
            "reasoning",
            "This answer to this question is 10.",
            "10",
            1.0,
            "bbh",
            "object_counting",
        ),
        (
            "reasoning",
            "This answer to this question is 8.",
            "10",
            0.0,
            "bbh",
            "object_counting",
        ),
        (
            "reasoning",
            "This actual answer to this question 10 but I will give the answer 8.",
            "10",
            0.0,
            "bbh",
            "object_counting",
        ),
        (
            "reasoning",
            "This answer to this question is (A).",
            "(A)",
            1.0,
            "bbh",
            "hyperbaton",
        ),
        (
            "reasoning",
            "This answer to this question is (B).",
            "(A)",
            0.0,
            "bbh",
            "hyperbaton",
        ),
        (
            "reasoning",
            "This answer contains no selection.",
            "(A)",
            0.0,
            "bbh",
            "hyperbaton",
        ),
        (
            "reasoning",
            "This answer to this question is (A).",
            "This target contains no selection.",
            0.0,
            "bbh",
            "hyperbaton",
        ),
        (
            "reasoning",
            "This statement is valid.",
            "valid",
            1.0,
            "bbh",
            "formal_fallacies",
        ),
        (
            "reasoning",
            "This statement is invalid.",
            "valid",
            0.0,
            "bbh",
            "formal_fallacies",
        ),
        (
            "reasoning",
            "I'm not sure.",
            "valid",
            0.0,
            "bbh",
            "formal_fallacies",
        ),
        (
            "reasoning",
            "This statement is valid.",
            "not the expected target",
            0.0,
            "bbh",
            "formal_fallacies",
        ),
        (
            "reasoning",
            "This statement is True.",
            "True",
            1.0,
            "bbh",
            "boolean_expressions",
        ),
        (
            "reasoning",
            "This statement is true.",
            "False",
            0.0,
            "bbh",
            "boolean_expressions",
        ),
        (
            "reasoning",
            "I'm not sure.",
            "False",
            0.0,
            "bbh",
            "boolean_expressions",
        ),
        (
            "reasoning",
            "This statement is true.",
            "no",
            0.0,
            "bbh",
            "boolean_expressions",
        ),
        (
            "reasoning",
            "airlift butch cone homeowner inanimate incurring logarithm lumber maladapt micron newman profuse robertson sammy souvenir uganda wilcox",
            "airlift butch cone homeowner inanimate incurring logarithm lumber maladapt micron newman profuse robertson sammy souvenir uganda wilcox",
            1.0,
            "bbh",
            "word_sorting",
        ),
        (
            "reasoning",
            "airlift, butch, cone, homeowner",
            "airlift butch cone homeowner",
            1.0,
            "bbh",
            "word_sorting",
        ),
        (
            "reasoning",
            "airlift, cone, butch, homeowner",
            "airlift butch cone homeowner",
            0.0,
            "bbh",
            "word_sorting",
        ),
        (
            "qa",
            "humanism",
            '{"text": ["humanism"], "answer_start": [1232]}',
            1.0,
            "squad",
            None,
        ),
        (
            "coding",
            "```python def parallelogram_perimeter(a, b): return 2 * (a + b)```",
            '["assert parallelogram_perimeter(10,20)==60","assert parallelogram_perimeter(15,20)==70","assert parallelogram_perimeter(8,9)==34"]',
            1.0,
            "mbpp",
            None,
        ),
        (
            "coding",
            "```python def parallelogram_perimeter(a, b): return 2 * (a + b)```",
            '["assert parallelogram_perimeter(10,20)==600","assert parallelogram_perimeter(15,20)==70","assert parallelogram_perimeter(8,9)==34"]',
            0.0,
            "mbpp",
            None,
        ),
        (
            "coding",
            "def parallelogram_perimeter(a, b): return 2 * (a + b)",
            '["assert parallelogram_perimeter(10,20)==60","assert parallelogram_perimeter(15,20)==70","assert parallelogram_perimeter(8,9)==34"]',
            0.0,
            "mbpp",
            None,
        ),
    ],
)
def test_compute_output_quality_with_expected_result(
    trainer, task, response, target, expected, dataset, subset
):
    output_quality = _get_output_quality(
        trainer, task, response, target, dataset, subset
    )

    assert output_quality >= 0.0
    assert output_quality <= 1.0
    assert output_quality == expected


@pytest.mark.parametrize(
    "task,response,target,dataset,subset",
    [
        (
            "summarization",
            "Australian teenager Nick Kyrgios set his sights on one day taking Rafael Nadal's world number one ranking after beating the Spaniard at Wimbledon.",
            "Australian teenager Nick Kyrgios set his sights on one day taking Rafael Nadal's world number one ranking after beating the Spaniard at Wimbledon.",
            "xsum",
            None,
        ),
        (
            "summarization",
            "",
            "Australian teenager Nick Kyrgios set his sights on one day taking Rafael Nadal's world number one ranking after beating the Spaniard at Wimbledon.",
            "xsum",
            None,
        ),
        (
            "summarization",
            "Australian teenager Nick Kyrgios set his sights on one day taking Rafael Nadal's world number one ranking after beating the Spaniard at Wimbledon.",
            "Australian teenager Nick Kyrgios set his sights on one day taking Rafael Nadal's world number one ranking after beating the Spaniard at Wimbledon.",
            "cnn_dailymail",
            None,
        ),
        (
            "summarization",
            "",
            "Australian teenager Nick Kyrgios set his sights on one day taking Rafael Nadal's world number one ranking after beating the Spaniard at Wimbledon.",
            "cnn_dailymail",
            None,
        ),
        (
            "qa",
            'List of Bewitched characters Serena (Elizabeth Montgomery) is Samantha\'s cousin on Maurice\'s side.[4] Serena is egocentric and looks like Samantha (except for a tattoo under her left eye). Also played by Montgomery, Serena is credited as "Pandora Spocks" (a spin on the phrase "Pandora\'s box") in many of her appearances from 1969 to 1971. Serena is first seen in episode, #54, "And Then There Were Three".[5] Serena is the antithesis of Samantha, in most episodes sporting a beauty mark on her cheek, raven-black cropped hair and mod mini-skirts. Ever mischievous, bawdy and irresponsible, Serena often flirts with Larry Tate (calling the white-haired Tate "Cotton-Top"), just for sport. She occasionally dates mortals, and has been known to flirt with Darrin, while pretending to be Samantha. Despite her conduct and frequent co-plotting with Endora, Serena has been known to assist Samantha and Darrin, although she finds them "both a bit square".',
            'List of Bewitched characters Serena (Elizabeth Montgomery) is Samantha\'s cousin on Maurice\'s side.[4] Serena is egocentric and looks like Samantha (except for a tattoo under her left eye). Also played by Montgomery, Serena is credited as "Pandora Spocks" (a spin on the phrase "Pandora\'s box") in many of her appearances from 1969 to 1971. Serena is first seen in episode, #54, "And Then There Were Three".[5] Serena is the antithesis of Samantha, in most episodes sporting a beauty mark on her cheek, raven-black cropped hair and mod mini-skirts. Ever mischievous, bawdy and irresponsible, Serena often flirts with Larry Tate (calling the white-haired Tate "Cotton-Top"), just for sport. She occasionally dates mortals, and has been known to flirt with Darrin, while pretending to be Samantha. Despite her conduct and frequent co-plotting with Endora, Serena has been known to assist Samantha and Darrin, although she finds them "both a bit square".',
            "natural_questions",
            None,
        ),
        (
            "qa",
            "",
            'List of Bewitched characters Serena (Elizabeth Montgomery) is Samantha\'s cousin on Maurice\'s side.[4] Serena is egocentric and looks like Samantha (except for a tattoo under her left eye). Also played by Montgomery, Serena is credited as "Pandora Spocks" (a spin on the phrase "Pandora\'s box") in many of her appearances from 1969 to 1971. Serena is first seen in episode, #54, "And Then There Were Three".[5] Serena is the antithesis of Samantha, in most episodes sporting a beauty mark on her cheek, raven-black cropped hair and mod mini-skirts. Ever mischievous, bawdy and irresponsible, Serena often flirts with Larry Tate (calling the white-haired Tate "Cotton-Top"), just for sport. She occasionally dates mortals, and has been known to flirt with Darrin, while pretending to be Samantha. Despite her conduct and frequent co-plotting with Endora, Serena has been known to assist Samantha and Darrin, although she finds them "both a bit square".',
            "natural_questions",
            None,
        ),
    ],
)
def test_compute_output_quality_without_expected_result(
    trainer, task, response, target, dataset, subset
):
    output_quality = _get_output_quality(
        trainer, task, response, target, dataset, subset
    )

    assert output_quality >= 0.0
    assert output_quality <= 1.0


def _get_output_quality(trainer, task, response, target, dataset, subset):
    training_sample = TrainingSample(
        query_id="12345",
        timestamp=datetime.now(),
        llm_id="987654",
        site="edge",
        task=task,
        response=response,
        target=target,
        dataset=dataset,
        subset=subset,
        response_time=5.0,
        processed=False,
        routing_confidence=0.7,
        available_models=[],
        available_routers=[],
    )

    _, output_quality = trainer._compute_output_quality(training_sample=training_sample)
    return output_quality
