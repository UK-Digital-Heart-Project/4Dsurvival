import json
from pathlib import Path
from pyhocon import ConfigTree, ConfigFactory


def get_conf(conf: ConfigTree, group: str = "", key: str = "", default=None):
    if group:
        key = ".".join([group, key])
    return conf.get(key, default)


class BaseConfig:
    GROUP = ""

    @classmethod
    def from_conf(cls, conf_path: Path):
        raise NotImplementedError("Must be implemented by subclasses")

    def save(self, output_dir: Path):
        assert output_dir.is_dir(), "output_dir has to be a directory."
        with open(str(output_dir.joinpath(self.__class__.__name__)) + ".json", "w") as file:
            json.dump(self.__dict__, file, indent=4)

    def to_dict(self):
        return self.__dict__


class ExperimentConfig(BaseConfig):
    def __init__(self, data_path: Path, output_dir: Path, n_evals: int, n_bootstraps: int, n_folds: int):
        self.data_path = data_path
        self.output_dir = output_dir
        self.n_evals = n_evals
        self.n_folds = n_folds
        self.n_bootstraps = n_bootstraps


class NNExperimentConfig(ExperimentConfig):
    GROUP = "experiment"

    def __init__(
            self, data_path: Path, output_dir: Path, n_evals: int, n_bootstraps: int, n_folds: int, batch_size: int,
            n_epochs: int, model_name: str
    ):
        super().__init__(
            data_path=data_path, output_dir=output_dir, n_evals=n_evals, n_bootstraps=n_bootstraps, n_folds=n_folds
        )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_name = model_name

    @classmethod
    def from_conf(cls, conf_path):
        conf = ConfigFactory.parse_file(str(conf_path))
        data_path = Path(get_conf(conf, group=cls.GROUP, key="data_path"))
        if get_conf(conf, group=cls.GROUP, key="output_dir") is None:
            output_dir = data_path.parent.joinpath("output")
        else:
            output_dir = Path(get_conf(conf, group=cls.GROUP, key="output_dir"))
        return cls(
            data_path=data_path,
            output_dir=output_dir,
            batch_size=get_conf(conf, group=cls.GROUP, key="batch_size", default=16),
            n_epochs=get_conf(conf, group=cls.GROUP, key="n_epochs", default=100),
            n_evals=get_conf(conf, group=cls.GROUP, key="n_evals", default=50),
            n_bootstraps=get_conf(conf, group=cls.GROUP, key="n_bootstraps", default=100),
            n_folds=get_conf(conf, group=cls.GROUP, key="n_folds", default=6),
            model_name=get_conf(conf, group=cls.GROUP, key="model_name", default="baseline_autoencoder")
        )


class CoxExperimentConfig(ExperimentConfig):
    def __init__(
            self, data_path: Path, output_dir: Path, n_evals: int, n_bootstraps: int, n_folds: int, penalty_exp: int
    ):
        super().__init__(
            data_path=data_path, output_dir=output_dir, n_evals=n_evals, n_bootstraps=n_bootstraps, n_folds=n_folds
        )
        self.penalty_exp = penalty_exp

    @classmethod
    def from_conf(cls, conf_path):
        conf = ConfigFactory.parse_file(str(conf_path))
        data_path = Path(get_conf(conf, group=cls.GROUP, key="data_path"))
        if get_conf(conf, group=cls.GROUP, key="output_dir") is None:
            output_dir = data_path.parent.joinpath("output")
        else:
            output_dir = Path(get_conf(conf, group=cls.GROUP, key="output_dir"))
        return cls(
            data_path=data_path,
            output_dir=output_dir,
            n_evals=get_conf(conf, group=cls.GROUP, key="n_evals", default=50),
            n_bootstraps=get_conf(conf, group=cls.GROUP, key="n_bootstraps", default=100),
            n_folds=get_conf(conf, group=cls.GROUP, key="n_folds", default=6),
            penalty_exp=get_conf(conf, group=cls.GROUP, key="penalty_exp", default=[-2, 1])
        )


class HypersearchConfig(BaseConfig):
    GROUP = "hypersearch"

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @classmethod
    def from_conf(cls, conf_path: Path):
        conf = ConfigFactory.parse_file(str(conf_path))
        conf = getattr(conf, cls.GROUP)
        return cls(**conf)


config = HypersearchConfig.from_conf(Path(__file__).parent.parent.joinpath("demo", "scripts", "default_validate_DL.conf"))
print(config)
