import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Set
import spacy

logger = logging.getLogger(__name__)

# Global variable to hold the spaCy model
nlp = None

def load_spacy_model():
    """Load spaCy model if not already loaded."""
    global nlp
    if nlp is None:
        try:
            logger.info("Loading spaCy model 'en_core_web_sm'...")
            nlp = spacy.load("en_core_web_sm") # disable=['parser', 'ner'] to speed up if just tagging
        except OSError:
            logger.warning("Model 'en_core_web_sm' not found. Please download it using: py -m spacy download en_core_web_sm")
            raise

# --- Custom Lists per User Requirements ---

# Words to REMOVE (Corporate Boilerplate)
EXCLUDE_WORDS = {
    'company', 'limited', 'ltd', 'incorporated', 'subsidiary', 'group', 
    'report', 'financial', 'year', 'fiscal', 'ended', 'march', 'date', 
    'page', 'table', 'figure'
}

# Words to KEEP (ESG Domain Specific) - forced inclusion even if they might be filtered otherwise
KEEP_WORDS = {
    'gigajoule', 'tco2e', 'kilolitre', 'biodiversity', 'paternity', 
    'maternity', 'grievance', 'fiduciary', 'vigil', 'committee'
}

def get_custom_stopwords() -> Set[str]:
    """
    Get set of custom stopwords including standard English and domain-specific exclude words.
    Does NOT include the KEEP_WORDS.
    """
    # Standard English stopwords
    standard_stopwords = nlp.Defaults.stop_words if nlp else set()
    if not standard_stopwords:
        # Fallback if nlp not loaded yet, though advanced_preprocess loads it
        standard_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
            'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'i', 'me', 'my', 'mine',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'down',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'not', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
    
    return standard_stopwords.union(EXCLUDE_WORDS)


def advanced_preprocess(text: str) -> str:
    """
    Advanced text preprocessing for ESG topic modeling.
    
    Steps:
    1. Clean text (lowercase, special chars).
    2. Tokenize and POS Tag using spaCy.
    3. Filter: Keep only NOUN, ADJ, PROPN.
    4. Remove EXCLUDE_WORDS.
    5. Ensure KEEP_WORDS are preserved.
    
    Args:
        text: Input text.
        
    Returns:
        Processed text string (space-separated tokens).
    """
    if not text:
        return ""
    
    # Ensure model is loaded
    load_spacy_model()
    
    # Basic cleaning first
    text = text.lower()
    # Remove emails/URLs
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove numbers (but keep tco2e etc which might have numbers mixed? regex below handles basic cleaning)
    # We'll rely on spaCy for tokenization primarily, but let's clear noise first.
    # Replace non-alphanumeric with space, but we need to be careful not to break "tco2e"
    # The requirement asks to keep "tco2e". It contains digits.
    # So we should probably NOT aggressively remove numbers if they are part of a target word.
    
    # Let's use spaCy directly on the relatively raw text to preserve token structure like "tco2e".
    # Just collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    doc = nlp(text)
    
    cleaned_tokens = []
    
    # Allowed POS tags
    allowed_pos = {'NOUN', 'PROPN', 'ADJ'}
    
    for token in doc:
        word = token.text.lower()
        lemma = token.lemma_.lower()
        
        # Check FORCE KEEP first
        if word in KEEP_WORDS or lemma in KEEP_WORDS:
            cleaned_tokens.append(word) # Keep original or lemma? Requirement implies keeping the concept.
            continue
            
        # Check EXCLUDE logic
        if word in EXCLUDE_WORDS or lemma in EXCLUDE_WORDS:
            continue
            
        # Check Stopwords matches
        if token.is_stop:
            continue
            
        # Check POS tags
        if token.pos_ in allowed_pos:
            # Check length to avoid noise like single letters not in keep list
            if len(word) > 2:
                cleaned_tokens.append(lemma) # Use lemma for topic clarity
                
    return ' '.join(cleaned_tokens)






def extract_text_from_json(json_path: Path) -> str:
    """
    Recursively extract all text from JSON structure.
    
    Extracts headings and content from all sections and subsections.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Full text string with all headings and content
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text_parts = []
    
    # Extract from structure recursively
    if 'structure' in data:
        text_parts.extend(_extract_from_structure(data['structure']))
    
    # Join with newlines
    full_text = '\n\n'.join(text_parts)
    
    return full_text


def _extract_from_structure(structure: List[Dict[str, Any]], level: int = 0) -> List[str]:
    """
    Recursively extract text from nested structure.
    
    Args:
        structure: List of content blocks (with heading, content, subsections)
        level: Current nesting level (for indentation)
        
    Returns:
        List of text strings (headings + content)
    """
    text_parts = []
    
    for block in structure:
        # Add heading
        heading = block.get('heading', '')
        if heading:
            text_parts.append(heading)
        
        # Add content paragraphs
        content = block.get('content', [])
        if content:
            if isinstance(content, list):
                text_parts.extend(content)
            elif isinstance(content, str):
                text_parts.append(content)
        
        # Recursively process subsections
        subsections = block.get('subsections', [])
        if subsections:
            text_parts.extend(_extract_from_structure(subsections, level + 1))
    
    return text_parts


def split_json_into_documents(
    json_data: Dict[str, Any], 
    strategy: str = "sections",
    min_length: int = 100
) -> List[Dict[str, Any]]:
    """
    Split JSON structure into documents for topic modeling.
    
    Args:
        json_data: Loaded JSON data dictionary
        strategy: How to split ("sections", "paragraphs", "both")
        min_length: Minimum character length for a document to be included
        
    Returns:
        List of dictionaries with keys: text, section_heading, section_level, section_path
    """
    documents = []
    
    if strategy == "sections":
        documents = _split_by_sections(json_data.get('structure', []), min_length)
    elif strategy == "paragraphs":
        documents = _split_by_paragraphs(json_data, min_length)
    elif strategy == "both":
        # Try sections first, fallback to paragraphs for empty sections
        documents = _split_by_sections(json_data.get('structure', []), min_length)
        if not documents:
            documents = _split_by_paragraphs(json_data, min_length)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'sections', 'paragraphs', or 'both'")
    
    # Filter by minimum length
    documents = [doc for doc in documents if len(doc['text'].strip()) >= min_length]
    
    logger.info(f"Split into {len(documents)} documents (strategy: {strategy}, min_length: {min_length})")
    
    return documents


def _split_by_sections(
    structure: List[Dict[str, Any]], 
    min_length: int,
    parent_path: str = "",
    level: int = 0
) -> List[Dict[str, Any]]:
    """
    Split JSON structure into documents based on sections.
    
    Each section (with heading + content) becomes one document.
    """
    documents = []
    
    for i, block in enumerate(structure):
        heading = block.get('heading', '')
        content = block.get('content', [])
        
        # Build section path
        section_path = f"{parent_path}/{heading}" if parent_path else heading
        
        # Combine heading and content
        text_parts = []
        if heading:
            text_parts.append(heading)
        
        if isinstance(content, list):
            text_parts.extend([p for p in content if p and p.strip()])
        elif isinstance(content, str) and content.strip():
            text_parts.append(content)
        
        # Join into single text
        text = '\n\n'.join(text_parts)
        
        # Only add if meets minimum length
        if len(text.strip()) >= min_length:
            documents.append({
                'text': text,
                'section_heading': heading,
                'section_level': block.get('level', level),
                'section_path': section_path
            })
        
        # Recursively process subsections
        subsections = block.get('subsections', [])
        if subsections:
            documents.extend(
                _split_by_sections(subsections, min_length, section_path, level + 1)
            )
    
    return documents


def _split_by_paragraphs(json_data: Dict[str, Any], min_length: int) -> List[Dict[str, Any]]:
    """
    Split JSON structure into documents based on paragraphs.
    
    Each paragraph becomes one document.
    """
    documents = []
    
    # Extract all paragraphs from structure
    structure = json_data.get('structure', [])
    paragraphs = _extract_paragraphs_recursive(structure)
    
    # Create document for each paragraph
    for i, para in enumerate(paragraphs):
        if len(para.strip()) >= min_length:
            documents.append({
                'text': para,
                'section_heading': f"Paragraph {i+1}",
                'section_level': 0,
                'section_path': f"paragraph_{i+1}"
            })
    
    return documents


def _extract_paragraphs_recursive(structure: List[Dict[str, Any]]) -> List[str]:
    """
    Recursively extract all paragraphs from structure.
    """
    paragraphs = []
    
    for block in structure:
        content = block.get('content', [])
        if isinstance(content, list):
            paragraphs.extend([p for p in content if p and p.strip()])
        elif isinstance(content, str) and content.strip():
            paragraphs.append(content)
        
        # Process subsections
        subsections = block.get('subsections', [])
        if subsections:
            paragraphs.extend(_extract_paragraphs_recursive(subsections))
    
    return paragraphs
