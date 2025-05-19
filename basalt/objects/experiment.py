from datetime import datetime
from ..ressources.monitor.experiment_types import Experiment as IExperiment

class Experiment:
    def __init__(self, experiment: IExperiment):
        self._experiment = experiment

    @property
    def id(self) -> str:
        return self._experiment.id

    @property
    def name(self) -> str:
        return self._experiment.name

    @property
    def feature_slug(self) -> str:
        return self._experiment.feature_slug

    @property
    def created_at(self) -> datetime:
        return self._experiment.created_at
