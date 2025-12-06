"""
Romanization to Hangul Jamo Mapping.

This module creates mappings between romanization patterns and actual Hangul Jamo components.
"""

# Romanization to 초성 (Initial Consonants) mapping
CHO_ROM_TO_JAMO = {
    "": "ㅇ",  # No initial = ㅇ (silent)
    "g": "ㄱ",
    "gg": "ㄲ",
    "n": "ㄴ",
    "d": "ㄷ",
    "dd": "ㄸ",
    "r": "ㄹ",
    "m": "ㅁ",
    "b": "ㅂ",
    "bb": "ㅃ",
    "s": "ㅅ",
    "ss": "ㅆ",
    "": "ㅇ",
    "j": "ㅈ",
    "jj": "ㅉ",
    "ch": "ㅊ",
    "k": "ㅋ",
    "t": "ㅌ",
    "p": "ㅍ",
    "h": "ㅎ",
}

# Romanization to 중성 (Medial Vowels) mapping
JUNG_ROM_TO_JAMO = {
    "a": "ㅏ",
    "ae": "ㅐ",
    "ya": "ㅑ",
    "yae": "ㅒ",
    "eo": "ㅓ",
    "e": "ㅔ",
    "yeo": "ㅕ",
    "ye": "ㅖ",
    "o": "ㅗ",
    "wa": "ㅘ",
    "wae": "ㅙ",
    "oe": "ㅚ",
    "yo": "ㅛ",
    "u": "ㅜ",
    "wo": "ㅝ",
    "we": "ㅞ",
    "wi": "ㅟ",
    "yu": "ㅠ",
    "eu": "ㅡ",
    "ui": "ㅢ",
    "i": "ㅣ",
}

# Romanization to 종성 (Final Consonants) mapping
JONG_ROM_TO_JAMO = {
    "": "",  # No final
    "k": "ㄱ",
    "n": "ㄴ",
    "l": "ㄹ",
    "m": "ㅁ",
    "ng": "ㅇ",
    "s": "ㅅ",
    "ss": "ㅆ",
    "t": "ㅌ",
}

# Reverse mappings (Jamo to romanization)
CHO_JAMO_TO_ROM = {v: k for k, v in CHO_ROM_TO_JAMO.items()}
JUNG_JAMO_TO_ROM = {v: k for k, v in JUNG_ROM_TO_JAMO.items()}
JONG_JAMO_TO_ROM = {v: k for k, v in JONG_ROM_TO_JAMO.items()}


def parse_romanized_character(rom_char):
    """
    Parse a romanized character into cho/jung/jong components.

    Args:
        rom_char: Romanized character (e.g., 'jeong', 'gwa', 'i')

    Returns:
        tuple: (cho_rom, jung_rom, jong_rom)

    Examples:
        jeong -> ('j', 'eo', 'ng')
        gwa -> ('g', 'wa', '')
        i -> ('', 'i', '')
    """
    # Special cases for class names from the dataset
    special_patterns = {
        # Characters with complex romanization
        "jeong": ("j", "eo", "ng"),
        "jeon": ("j", "eo", "n"),
        "jeok": ("j", "eo", "k"),
        "gyeong": ("g", "yeo", "ng"),
        "seong": ("s", "eo", "ng"),
        "yeo": ("y", "eo", ""),
        "yeon": ("y", "eo", "n"),
        "choe": ("ch", "o", "e"),  # Actually 최
        "geos": ("g", "eo", "s"),
        "deul": ("d", "eu", "l"),
        "reul": ("r", "eu", "l"),
        "neun": ("n", "eu", "n"),
        "eun": ("", "eu", "n"),
        "eul": ("", "eu", "l"),
        "eui": ("", "eu", "i"),
        "seu": ("s", "eu", ""),
        "geu": ("g", "eu", ""),
        "iss": ("", "i", "ss"),
        "dong": ("d", "o", "ng"),
        "gong": ("g", "o", "ng"),
        "jang": ("j", "a", "ng"),
        "sang": ("s", "a", "ng"),
        "yong": ("y", "o", "ng"),
        "won": ("w", "o", "n"),
        "guk": ("g", "u", "k"),
        "bak": ("b", "a", "k"),
        "gim": ("g", "i", "m"),
        "han": ("h", "a", "n"),
        "hae": ("h", "a", "e"),
        "dae": ("d", "a", "e"),
        "gye": ("g", "ye", ""),
        "hwa": ("h", "wa", ""),
        "gwa": ("g", "wa", ""),
        # Simple patterns
        "a": ("", "a", ""),
        "e": ("", "e", ""),
        "eo": ("", "eo", ""),
        "eu": ("", "eu", ""),
        "i": ("", "i", ""),
        "o": ("", "o", ""),
        "u": ("", "u", ""),
        # Consonant + vowel patterns
        "bo": ("b", "o", ""),
        "bu": ("b", "u", ""),
        "da": ("d", "a", ""),
        "do": ("d", "o", ""),
        "ga": ("g", "a", ""),
        "gi": ("g", "i", ""),
        "go": ("g", "o", ""),
        "gu": ("g", "u", ""),
        "ha": ("h", "a", ""),
        "ja": ("j", "a", ""),
        "je": ("j", "e", ""),
        "ji": ("j", "i", ""),
        "jo": ("j", "o", ""),
        "ju": ("j", "u", ""),
        "na": ("n", "a", ""),
        "ra": ("r", "a", ""),
        "ri": ("r", "i", ""),
        "ro": ("r", "o", ""),
        "sa": ("s", "a", ""),
        "seo": ("s", "eo", ""),
        "si": ("s", "i", ""),
        "so": ("s", "o", ""),
        "su": ("s", "u", ""),
        "wi": ("w", "i", ""),
        "in": ("", "i", "n"),
        "il": ("", "i", "l"),
    }

    if rom_char in special_patterns:
        return special_patterns[rom_char]

    # Default: return as-is (will need manual fixing)
    print(f"Warning: No pattern found for '{rom_char}'")
    return ("", rom_char, "")


def romanization_to_jamo(rom_char):
    """
    Convert romanized character to actual Hangul Jamo components.
    
    Args:
        rom_char: Romanized character
    
    Returns:
        tuple: (초성, 중성, 종성) in Hangul Jamo
    """
    cho_rom, jung_rom, jong_rom = parse_romanized_character(rom_char)
    
    cho_jamo = CHO_ROM_TO_JAMO.get(cho_rom, "")
    jung_jamo = JUNG_ROM_TO_JAMO.get(jung_rom, "")
    jong_jamo = JONG_ROM_TO_JAMO.get(jong_rom, "")
    
    return cho_jamo, jung_jamo, jong_jamo


def romanization_to_romanized_jamos(rom_char):
    """
    Convert romanized character to romanized Jamo component names.
    
    This returns the romanization strings themselves, matching the
    Jamo classifier's class names (e.g., 'g', 'a', 'ng').
    
    Args:
        rom_char: Romanized character
    
    Returns:
        tuple: (cho_rom, jung_rom, jong_rom) as romanization strings
    """
    cho_rom, jung_rom, jong_rom = parse_romanized_character(rom_char)
    return cho_rom, jung_rom, jong_rom


if __name__ == "__main__":
    # Test with known class names
    test_chars = [
        "jeong",
        "jeon",
        "jeok",
        "gyeong",
        "yeo",
        "eo",
        "i",
        "ji",
        "gi",
        "deul",
        "reul",
    ]

    print("Romanization to Jamo Mapping Test:")
    print("=" * 60)
    for char in test_chars:
        cho, jung, jong = romanization_to_jamo(char)
        print(f"{char:10s} -> 초성:{cho} 중성:{jung} 종성:{jong}")
