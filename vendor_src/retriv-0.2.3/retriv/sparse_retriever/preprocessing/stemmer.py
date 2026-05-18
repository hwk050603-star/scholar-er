from functools import partial
from typing import Union

import nltk

from .utils import identity_function

try:
    from krovetzstemmer import Stemmer as KrovetzStemmer
except ImportError:
    KrovetzStemmer = None

try:
    from Stemmer import Stemmer as SnowballStemmer
except ImportError:
    SnowballStemmer = None


def _missing_stemmer(name: str):
    def _raise(_: str) -> str:
        raise ImportError(
            f"Stemmer '{name}' requires an optional dependency that is not installed."
        )

    return _raise


stemmers_dict = {
    "porter": partial(nltk.stem.PorterStemmer().stem),
    "lancaster": partial(nltk.stem.LancasterStemmer().stem),
    "arlstem": partial(nltk.stem.ARLSTem().stem),
    "arlstem2": partial(nltk.stem.ARLSTem2().stem),
    "cistem": partial(nltk.stem.Cistem().stem),
    "isri": partial(nltk.stem.ISRIStemmer().stem),
    "english": partial(nltk.stem.SnowballStemmer("english").stem),
}

if KrovetzStemmer is not None:
    stemmers_dict["krovetz"] = partial(KrovetzStemmer())
else:
    stemmers_dict["krovetz"] = _missing_stemmer("krovetz")

if SnowballStemmer is not None:
    for language in [
        "arabic",
        "basque",
        "catalan",
        "danish",
        "dutch",
        "finnish",
        "french",
        "german",
        "greek",
        "hindi",
        "hungarian",
        "indonesian",
        "irish",
        "italian",
        "lithuanian",
        "nepali",
        "norwegian",
        "portuguese",
        "romanian",
        "russian",
        "spanish",
        "swedish",
        "tamil",
        "turkish",
    ]:
        stemmers_dict[language] = partial(SnowballStemmer(language).stemWord)
else:
    for language in [
        "arabic",
        "basque",
        "catalan",
        "danish",
        "dutch",
        "finnish",
        "french",
        "german",
        "greek",
        "hindi",
        "hungarian",
        "indonesian",
        "irish",
        "italian",
        "lithuanian",
        "nepali",
        "norwegian",
        "portuguese",
        "romanian",
        "russian",
        "spanish",
        "swedish",
        "tamil",
        "turkish",
    ]:
        stemmers_dict[language] = _missing_stemmer(language)


def _get_stemmer(stemmer: str) -> callable:
    assert stemmer.lower() in stemmers_dict, f"Stemmer {stemmer} not supported."
    return stemmers_dict[stemmer.lower()]


def get_stemmer(stemmer: Union[str, callable, bool]) -> callable:
    if isinstance(stemmer, str):
        return _get_stemmer(stemmer)
    elif callable(stemmer):
        return stemmer
    elif stemmer is None:
        return identity_function
    else:
        raise NotImplementedError
