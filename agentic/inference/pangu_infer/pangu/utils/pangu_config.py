from types import SimpleNamespace


class PanguConfig(SimpleNamespace):
    """
    Configuration container for patches.
    Attributes are accessible via dot notation and also via [].
    Nested dicts/lists are automatically wrapped.
    """

    def __init__(self, config_dict: dict):
        # Recursively wrap dicts and lists
        def wrap(obj):
            if isinstance(obj, dict):
                return PanguConfig(obj)
            if isinstance(obj, list):
                return [wrap(v) for v in obj]
            return obj

        wrapped = {k: wrap(v) for k, v in config_dict.items()}
        super().__init__(**wrapped)

    def __getitem__(self, key):
        # Return attribute or None if missing
        return self.__dict__.get(key, None)

    def __getattr__(self, key):
        # Override to return None for missing attributes
        return None

    def get(self, key, default=None):
        """
        Return the value for 'key' if present, else 'default'.

        Args:
            key (str): Configuration attribute name.
            default: Value to return if key is not found.

        Returns:
            The attribute value or default.
        """
        return self.__dict__.get(key, default)
