"""
Flatten Downloads - Move PDFs from nested folder structure to flat structure.

This script moves all PDFs from brsr_reports/downloads/{company}/{year}/*.pdf
to brsr_reports/downloads/*.pdf (flat structure).
"""
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def flatten_downloads_folder(downloads_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Move all PDFs from nested folder structure to flat structure.
    
    Args:
        downloads_dir: Path to downloads directory (e.g., brsr_reports/downloads)
        dry_run: If True, only report what would be moved without actually moving
        
    Returns:
        Tuple of (files_moved, files_skipped)
    """
    downloads_dir = Path(downloads_dir)
    
    if not downloads_dir.exists():
        logger.error(f"Downloads directory does not exist: {downloads_dir}")
        return 0, 0
    
    logger.info(f"Scanning downloads directory: {downloads_dir}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'MOVE FILES'}")
    logger.info("=" * 60)
    
    files_moved = 0
    files_skipped = 0
    files_to_move: List[Tuple[Path, Path]] = []
    
    # Find all PDF files in nested folders (but not in root)
    for pdf_file in downloads_dir.rglob("*.pdf"):
        # Skip PDFs already in the root downloads folder
        if pdf_file.parent == downloads_dir:
            logger.debug(f"Skipping (already in root): {pdf_file.name}")
            continue
        
        # Calculate destination (same filename in root)
        dest_path = downloads_dir / pdf_file.name
        
        files_to_move.append((pdf_file, dest_path))
    
    logger.info(f"Found {len(files_to_move)} PDF(s) in nested folders")
    logger.info("")
    
    # Process each file
    for src_path, dest_path in files_to_move:
        rel_path = src_path.relative_to(downloads_dir.parent)
        
        if dest_path.exists():
            logger.warning(f"⚠ SKIP: {src_path.name}")
            logger.warning(f"   Source: {rel_path}")
            logger.warning(f"   Destination already exists: {dest_path.name}")
            files_skipped += 1
            continue
        
        if dry_run:
            logger.info(f"Would move: {rel_path} -> {dest_path.name}")
            files_moved += 1
        else:
            try:
                shutil.move(str(src_path), str(dest_path))
                logger.info(f"✓ Moved: {rel_path} -> {dest_path.name}")
                files_moved += 1
            except Exception as e:
                logger.error(f"✗ Failed to move {rel_path}: {e}")
                files_skipped += 1
    
    # Clean up empty folders (after moving files)
    if not dry_run and files_moved > 0:
        logger.info("")
        logger.info("Cleaning up empty folders...")
        folders_removed = 0
        
        # Find all directories (except root) and remove if empty
        for folder in sorted(downloads_dir.rglob("*"), reverse=True):
            if folder.is_dir() and folder != downloads_dir:
                try:
                    # Try to remove if empty
                    folder.rmdir()
                    logger.debug(f"Removed empty folder: {folder.relative_to(downloads_dir.parent)}")
                    folders_removed += 1
                except OSError:
                    # Folder not empty or other error - skip
                    pass
        
        if folders_removed > 0:
            logger.info(f"Removed {folders_removed} empty folder(s)")
    
    return files_moved, files_skipped


def main():
    """Main entry point."""
    import sys
    
    # Get downloads directory path
    # Script is in tests/, so go up to pdf-to-structured-reports/
    script_dir = Path(__file__).parent  # tests/
    project_dir = script_dir.parent  # pdf-to-structured-reports/
    # Downloads folder is inside pdf-to-structured-reports/brsr_reports/downloads/
    downloads_dir = project_dir / "brsr_reports" / "downloads"
    
    # Alternative: use config if available
    try:
        import sys
        sys.path.insert(0, str(project_dir))
        from config.config import DOWNLOAD_BASE_DIR
        downloads_dir = DOWNLOAD_BASE_DIR
    except ImportError:
        pass  # Use calculated path
    
    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    
    logger.info("=" * 60)
    logger.info("FLATTEN DOWNLOADS FOLDER")
    logger.info("=" * 60)
    logger.info(f"Downloads directory: {downloads_dir}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 60)
    logger.info("")
    
    if not dry_run:
        response = input("This will move all PDFs to flat structure. Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("Cancelled by user")
            return
    
    files_moved, files_skipped = flatten_downloads_folder(downloads_dir, dry_run=dry_run)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Files moved: {files_moved}")
    logger.info(f"Files skipped: {files_skipped}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
