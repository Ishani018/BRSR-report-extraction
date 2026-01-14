"""
Quality Assessment Script for BRSR Extractions

Analyzes extracted DOCX and JSON files to assess:
- Text coherence and readability
- Jumbled text detection
- Character encoding issues
- Content completeness
- BRSR-specific content detection
- Statistical quality metrics
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

from docx import Document

# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.config import OUTPUT_BASE_DIR, LOG_FORMAT, LOG_LEVEL
import logging

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def detect_jumbled_text(text: str) -> Dict:
    """Detect jumbled text patterns."""
    issues = []
    non_word_ratio = 0
    
    # Check for excessive random characters (not words/numbers)
    non_word_chars = len(re.findall(r'[^\w\s.,;:!?()\[\]{}"\'-]', text))
    total_chars = len(text)
    if total_chars > 0:
        non_word_ratio = non_word_chars / total_chars
        if non_word_ratio > 0.1:  # More than 10% non-word chars
            issues.append(f"High ratio of non-word characters: {non_word_ratio:.1%}")
    
    # Check for excessive single-character words (might indicate jumbled text)
    words = text.split()
    if len(words) > 100:
        single_char_words = sum(1 for w in words if len(w) == 1 and w.isalpha())
        if single_char_words / len(words) > 0.15:  # More than 15% single char
            issues.append(f"High ratio of single-character words: {single_char_words/len(words):.1%}")
    
    # Check for words with no vowels (likely OCR errors or gibberish)
    words_with_no_vowels = sum(1 for w in words if w.isalpha() and len(w) > 2 and not re.search(r'[aeiouAEIOU]', w))
    if len(words) > 0 and words_with_no_vowels / len(words) > 0.05:
        issues.append(f"Many words without vowels: {words_with_no_vowels} ({words_with_no_vowels/len(words):.1%})")
    
    # Check for excessive uppercase/lowercase mixing (might indicate encoding issues)
    if len(text) > 100:
        mixed_case_words = sum(1 for w in words if w.isalpha() and len(w) > 3 
                              and not (w.isupper() or w.islower() or w.istitle()))
        if len(words) > 0 and mixed_case_words / len(words) > 0.3:
            issues.append(f"Excessive mixed case words: {mixed_case_words/len(words):.1%}")
    
    return {
        "has_issues": len(issues) > 0,
        "issues": issues,
        "non_word_ratio": non_word_ratio if total_chars > 0 else 0
    }


def check_text_coherence(text: str) -> Dict:
    """Check text coherence and readability."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    metrics = {
        "total_words": len(words),
        "total_sentences": len(sentences),
        "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0
    }
    
    # Check for reasonable sentence structure
    very_short_sentences = sum(1 for s in sentences if len(s.split()) < 3)
    very_long_sentences = sum(1 for s in sentences if len(s.split()) > 50)
    
    coherence_issues = []
    if sentences:
        if very_short_sentences / len(sentences) > 0.3:
            coherence_issues.append(f"Too many very short sentences: {very_short_sentences/len(sentences):.1%}")
        if very_long_sentences / len(sentences) > 0.2:
            coherence_issues.append(f"Too many very long sentences: {very_long_sentences/len(sentences):.1%}")
    
    # Check for paragraph structure
    paragraphs = text.split('\n\n')
    avg_para_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
    
    metrics["coherence_issues"] = coherence_issues
    metrics["avg_paragraph_length"] = avg_para_length
    
    return metrics


def check_brsr_content(text: str) -> Dict:
    """Check for BRSR-specific content indicators."""
    text_lower = text.lower()
    
    indicators = {
        "section_a": bool(re.search(r'\bsection\s+a\b', text_lower)),
        "section_b": bool(re.search(r'\bsection\s+b\b', text_lower)),
        "principle": bool(re.search(r'\bprinciple\b', text_lower)),
        "business_responsibility": bool(re.search(r'\bbusiness\s+responsibility\b', text_lower)),
        "sustainability": bool(re.search(r'\bsustainability\b', text_lower)),
        "brsr": bool(re.search(r'\bbrsr\b', text_lower, re.I)),
        "ngrbc": bool(re.search(r'\bngrbc\b', text_lower, re.I)),
        "general_information": bool(re.search(r'\bgeneral\s+information\b', text_lower)),
        "listed_entity": bool(re.search(r'\blisted\s+entity\b', text_lower)),
    }
    
    found_count = sum(indicators.values())
    expected_min = 5  # At least 5 indicators should be present for BRSR
    
    return {
        "indicators_found": found_count,
        "indicators": indicators,
        "is_likely_brsr": found_count >= expected_min
    }


def check_encoding_issues(text: str) -> Dict:
    """Check for encoding/character issues."""
    issues = []
    
    # Check for NULL bytes
    if '\x00' in text:
        issues.append("NULL bytes found")
    
    # Check for other control characters (except common ones like \n, \t, \r)
    control_chars = re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', text)
    if control_chars:
        unique_control = set(control_chars)
        issues.append(f"Control characters found: {len(unique_control)} unique types")
    
    # Check for replacement characters (indicates encoding issues)
    if '\ufffd' in text:
        replacement_count = text.count('\ufffd')
        issues.append(f"Replacement characters () found: {replacement_count}")
    
    # Check for unusual unicode patterns
    unusual_chars = re.findall(r'[\u2000-\u206F\u2070-\u209F\u20A0-\u20CF]', text)
    if unusual_chars:
        issues.append(f"Unusual Unicode characters found: {len(set(unusual_chars))} types")
    
    return {
        "has_issues": len(issues) > 0,
        "issues": issues
    }


def sample_text_quality(text: str, num_samples: int = 5) -> List[Dict]:
    """Sample text from different parts to check quality."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if not paragraphs:
        return []
    
    samples = []
    # Sample from beginning, middle, and end
    sample_indices = [0, len(paragraphs)//4, len(paragraphs)//2, 3*len(paragraphs)//4, len(paragraphs)-1]
    sample_indices = [i for i in sample_indices if i < len(paragraphs)][:num_samples]
    
    for idx in sample_indices:
        para = paragraphs[idx]
        words = para.split()
        sentences = re.split(r'[.!?]+', para)
        
        samples.append({
            "paragraph_index": idx,
            "char_count": len(para),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "preview": para[:200] + "..." if len(para) > 200 else para
        })
    
    return samples


def assess_docx_quality(docx_path: Path) -> Dict:
    """Assess quality of a DOCX file."""
    logger.info(f"Assessing: {docx_path.name}")
    
    try:
        doc = Document(docx_path)
        full_text = "\n\n".join([para.text for para in doc.paragraphs])
        
        if not full_text.strip():
            return {
                "status": "error",
                "error": "Empty document"
            }
        
        # Run all quality checks
        jumbled = detect_jumbled_text(full_text)
        coherence = check_text_coherence(full_text)
        brsr_content = check_brsr_content(full_text)
        encoding = check_encoding_issues(full_text)
        samples = sample_text_quality(full_text)
        
        # Calculate overall quality score (0-100)
        quality_score = 100
        
        # Penalize for issues
        if jumbled["has_issues"]:
            quality_score -= 20
        if encoding["has_issues"]:
            quality_score -= 15
        if not brsr_content["is_likely_brsr"]:
            quality_score -= 10
        if coherence.get("coherence_issues"):
            quality_score -= 10 * len(coherence["coherence_issues"])
        
        # Ensure score is between 0-100
        quality_score = max(0, min(100, quality_score))
        
        # Determine quality rating
        if quality_score >= 90:
            rating = "excellent"
        elif quality_score >= 75:
            rating = "good"
        elif quality_score >= 60:
            rating = "fair"
        elif quality_score >= 40:
            rating = "poor"
        else:
            rating = "very_poor"
        
        return {
            "status": "success",
            "filename": docx_path.name,
            "quality_score": quality_score,
            "quality_rating": rating,
            "total_characters": len(full_text),
            "total_words": len(full_text.split()),
            "jumbled_text": jumbled,
            "coherence": coherence,
            "brsr_content": brsr_content,
            "encoding": encoding,
            "text_samples": samples
        }
        
    except Exception as e:
        logger.error(f"Error assessing {docx_path.name}: {e}")
        return {
            "status": "error",
            "filename": docx_path.name,
            "error": str(e)
        }


def assess_all_extractions(output_dir: Path = OUTPUT_BASE_DIR) -> None:
    """Assess quality of all extracted files."""
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return
    
    docx_files = sorted(output_dir.glob("*.docx"))
    
    if not docx_files:
        logger.warning(f"No DOCX files found in {output_dir}")
        return
    
    logger.info(f"Found {len(docx_files)} DOCX file(s) to assess")
    logger.info("=" * 80)
    
    results = []
    
    for docx_path in docx_files:
        assessment = assess_docx_quality(docx_path)
        results.append(assessment)
    
    # Summary
    successful = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") == "error"]
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("QUALITY ASSESSMENT SUMMARY")
    logger.info("=" * 80)
    
    if successful:
        scores = [r["quality_score"] for r in successful]
        ratings = Counter(r["quality_rating"] for r in successful)
        
        logger.info(f"Total assessed: {len(successful)}")
        logger.info(f"Average quality score: {sum(scores)/len(scores):.1f}/100")
        logger.info("")
        logger.info("Quality distribution:")
        for rating in ["excellent", "good", "fair", "poor", "very_poor"]:
            count = ratings.get(rating, 0)
            if count > 0:
                logger.info(f"  {rating.capitalize()}: {count} ({count/len(successful)*100:.1f}%)")
        
        logger.info("")
        logger.info("Files with issues:")
        for r in successful:
            if r["quality_score"] < 70 or r["jumbled_text"]["has_issues"] or r["encoding"]["has_issues"]:
                logger.info(f"  ⚠ {r['filename']}: Score={r['quality_score']}/100 ({r['quality_rating']})")
                if r["jumbled_text"]["has_issues"]:
                    logger.info(f"     - Jumbled text issues: {', '.join(r['jumbled_text']['issues'])}")
                if r["encoding"]["has_issues"]:
                    logger.info(f"     - Encoding issues: {', '.join(r['encoding']['issues'])}")
                if not r["brsr_content"]["is_likely_brsr"]:
                    logger.info(f"     - Missing BRSR content (only {r['brsr_content']['indicators_found']}/9 indicators found)")
    
    if errors:
        logger.info("")
        logger.info(f"Files with errors: {len(errors)}")
        for r in errors:
            logger.info(f"  ✗ {r['filename']}: {r.get('error', 'Unknown error')}")
    
    # Save detailed report
    report_path = output_dir / "quality_assessment_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "assessment_date": str(Path(__file__).stat().st_mtime),
            "total_files": len(results),
            "successful": len(successful),
            "errors": len(errors),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info(f"Detailed report saved to: {report_path}")
    logger.info("Done!")


if __name__ == "__main__":
    assess_all_extractions()
