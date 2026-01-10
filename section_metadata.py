"""
Data structures for section metadata and boundaries.
"""
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class SectionType(Enum):
    """Enumeration of supported section types."""
    MDNA = "mdna"
    LETTER_TO_STAKEHOLDERS = "letter_to_stakeholders"
    UNKNOWN = "unknown"


@dataclass
class TextBlock:
    """
    Represents a text block extracted from PDF with layout metadata.
    """
    text: str
    page_number: int
    font_size: float
    y_position: float
    x_position: float
    bbox: tuple  # (x0, y0, x1, y1)
    
    @property
    def normalized_text(self) -> str:
        """Return lowercase text with punctuation removed."""
        import re
        text = self.text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text
    
    @property
    def line_length(self) -> int:
        """Return character count of the text."""
        return len(self.text)


@dataclass
class SectionBoundary:
    """
    Represents detected section boundaries with confidence score.
    """
    section_type: SectionType
    start_page: int
    end_page: Optional[int]
    confidence: float
    start_heading: str
    detection_method: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "section_type": self.section_type.value,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "confidence": round(self.confidence, 3),
            "start_heading": self.start_heading,
            "detection_method": self.detection_method
        }


@dataclass
class SectionContent:
    """
    Represents extracted section content.
    """
    section_type: SectionType
    start_page: int
    end_page: int
    text: str
    character_count: int
    page_count: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "section_type": self.section_type.value,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "character_count": self.character_count,
            "page_count": self.page_count
        }


# Canonical keyword patterns for section detection
SECTION_KEYWORDS = {
    SectionType.MDNA: [
        "management discussion and analysis",
        "management's discussion and analysis",
        "managements discussion and analysis",
        "md&a",
        "md a",
        "mda",
        "financial review"
    ],
    SectionType.LETTER_TO_STAKEHOLDERS: [
        "letter to stakeholders",
        "letter to shareholders",
        "chairman's letter",
        "chairmans letter",
        "chairman's message",
        "chairmans message",
        "ceo message",
        "ceo's message",
        "message from the chairman",
        "message from the ceo",
        "president's message",
        "letter from the chairman",
        "letter from the ceo",
        "dear stakeholders",
        "dear shareholders"
    ]
}

# Keywords that indicate section end
SECTION_END_KEYWORDS = [
    "financial statements",
    "consolidated financial statements",
    "notes to financial statements",
    "auditor's report",
    "auditors report",
    "independent auditor",
    "balance sheet",
    "income statement",
    "cash flow statement",
    "statement of financial position"
]
