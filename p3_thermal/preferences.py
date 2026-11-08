"""Small, validated user preferences stored separately from imported projects."""

import json

from .palettes import default_library_path, write_json


class Preferences:
    def __init__(self, path=None, model="p3"):
        self.path = path or default_library_path().with_name(f"settings-{model}.json")
        self.mirror = False
        self.remember = True
        self.language = "en"
        self.error = None
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or type(data.get("mirror")) is not bool:
                    raise ValueError("Invalid mirror preference")
                self.language = data.get("language", "en")
                if self.language not in ("en", "pl"):
                    raise ValueError("Unsupported language")
                self.mirror = data["mirror"]
                self.remember = data.get("remember_mirror", True) is True
        except (OSError, ValueError) as exc:
            self.error = str(exc)

    def save(self, mirror, remember=True, language=None):
        if self.error:
            raise ValueError(self.error)
        language = self.language if language is None else language
        if language not in ("en", "pl"):
            raise ValueError("Unsupported language")
        value = bool(mirror) if remember else False
        write_json(
            self.path,
            {
                "version": 1,
                "mirror": value,
                "remember_mirror": bool(remember),
                "language": language,
            },
        )
        self.language = language
        self.mirror = value
        self.remember = bool(remember)
