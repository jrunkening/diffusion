from pathlib import Path
import torchvision.datasets as datasets


DATA_PATH = Path(__file__).parent.joinpath("../data")

celeba = datasets.CelebA(DATA_PATH, "all", download=True)
