from dagster import Definitions, FilesystemIOManager

from dagster_components.assets import generate_data
from dagster_components.jobs import evaluate


io_manager = FilesystemIOManager()


defs = Definitions(
    assets=[generate_data],
    jobs=[evaluate],
    resources={
        "io_manager": io_manager,
    },
)
