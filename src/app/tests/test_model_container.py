from unittest import TestCase
import os
import numpy as np
from typing import List, Dict
from ..pipeline.model import ModelContainer

class ModelContainerTestCase(TestCase):

    def setUp(self):
        self.datadir: str = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
        self.log: bool = False
        self.filename:str = "initial_model"

        self.model_container = ModelContainer(self.datadir, log=self.log)

    def tearDown(self):
        pass


class TestModelContainerSingleton(ModelContainerTestCase):

    def test_load(self):
        # load data
        self.model_container.load(self.filename)

        scores: Dict[str, float] = self.model_container.score()
        print(scores)
