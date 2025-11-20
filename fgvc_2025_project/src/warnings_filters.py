import warnings


def install_warning_filters():
    # Silence common pydantic dataclasses warnings (benign) seen during imports of third-party libs
    try:
        from pydantic.dataclasses import UnsupportedFieldAttributeWarning  # type: ignore
        warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
    except Exception:
        # pydantic not available or structure changed; best-effort regex fallback
        warnings.filterwarnings(
            "ignore",
            message=r".*UnsupportedFieldAttributeWarning:.*'repr' attribute.*",
            category=UserWarning,
        )

    # Avoid noisy deprecation notices during training loops
    warnings.filterwarnings("ignore", category=DeprecationWarning)

