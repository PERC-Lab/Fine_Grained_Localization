
from dynaconf import Dynaconf

dummy_config = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/foo_dummy_config.toml'],
)

collection_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/basic_config.toml',
                    '.config/new_config_practice_collection.toml'],
    merge_enabled=True
)

other_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/basic_config.toml',
                    '.config/new_config_practice_other.toml'],
    merge_enabled=True
)

sharing_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/basic_config.toml',
                    '.config/new_config_practice_sharing.toml'],
    merge_enabled=True
)

processing_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/basic_config.toml',
                    '.config/new_config_practice_processing.toml'],
    merge_enabled=True
)

functionality_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/basic_config.toml',
                    '.config/new_config_purpose_functionality.toml'],
    merge_enabled=True
)

purpose_other_settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/basic_config.toml',
                    '.config/new_config_purpose_other.toml'],
    merge_enabled=True
)

purpose_advertisement = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/basic_config.toml',
                    '.config/new_config_purpose_advertisement.toml'],
    merge_enabled=True
)

purpose_analytics = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['.config/basic_config.toml',
                    '.config/new_config_purpose_analytics.toml'],
    merge_enabled=True
)
