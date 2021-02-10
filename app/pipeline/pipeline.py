import os
from .model import ModelContainer

pipelinepath: str = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.abspath(os.path.join(pipelinepath, '..', 'data'))


def train() -> ModelContainer:
    model_container = ModelContainer(datadir, log=False)
    print(model_container.datadir)
    model_container.train("initial_model")
    model_container.score()
    return model_container


def load(filename: str) -> ModelContainer:
    model_container = ModelContainer(datadir, log=False)
    print(model_container.datadir)
    model_container.load(filename)
    model_container.score()
    return model_container


if __name__ == "__main__":
    print(datadir)
    load("initial_model")