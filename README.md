# EcoRoute

EcoRoute is a multi-task Large Language Model routing framwork. It employs a collection of expert SLMs and LLMs to solve for different tasks in the Edge/Cloud-Continuum. It uses a Multi-Armed-Bandit approach to maintain high output accuracy while reducing response time and energy consumption, and balancing the workflow among the models. 

## Quickstart

### Local deployment

Use `docker compose -f docker-compose-local.yml up --build`

This starts 

- two NATS Jetstream clusters with a single server each
- two routers (one per cluster)
- two inference engines (one per cluster)

The inference engines only return dummy responses and do not read any energy consumption because no actual LLM is hosted locally. The number of inference engines can be scaled with e.g. `--scale=inference-nc=3`.

### Deployment on Chameleon Cloud

To deploy EcoRoute on Chameleon Cloud and reproduce the evaluation results, follow the steps in [Deploy](deploy/README.md).

## Project structure

```
.
├── .github
│   └── workflows
│       └── tests.yml
├── config
│   ├── inference
│   │   └── config.yml
│   ├── nats
│   │   ├── server-edge.conf
│   │   ├── server-tacc.conf
│   │   └── server-uc.conf
│   └── router
│       ├── bounds.yml
│       └── config.yml
├── data
│   ├── query-squad.json
│   └── ...
├── src
│   ├── inference
│   │   ├── inference
│   │   │   ├── config.py
│   │   │   ├── energy_tracker.py
│   │   │   └── llm.py
│   │   ├── Dockerfile
│   │   ├── Dockerfile.dependencies
│   │   ├── Dockerfile.local
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── router
│   │   ├── router
│   │   │   ├── app.py
│   │   │   ├── classification.py
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   ├── eval_task_classifier.py
│   │   │   ├── jetstream.py
│   │   │   ├── metrics.py
│   │   │   ├── models.py
│   │   │   ├── router.py
│   │   │   ├── routing.py
│   │   │   └── trainer.py
│   │   ├── Dockerfile
│   │   ├── Dockerfile.dependencies
│   │   ├── Dockerfile.local
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── docker-compose-local.yml
│   └── docker-compose.yml
├── test
│   └── unit
│       └── router
│           ├── test_routing.py
│           └── test_trainer.py
└── utils
    └── ...
```
