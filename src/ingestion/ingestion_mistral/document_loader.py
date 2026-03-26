"""Unified document loader with automatic OCR detection using Azure-hosted Mistral OCR."""

import os
import logging

from pathlib import Path
from typing import Dict, Any, List, Optional
from .pdf_extractor import PDFExtractor
from .ocr_engine import OCREngine, OcrMode

logger = logging.getLogger(__name__)

# File extensions recognised by each handler
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml"}

class DocumentLoader:
    """
    Unified document loading with automatic format detection and OCR.

    Decision logic
    ──────────────
    • Image files  → always OCR
    • PDF files    → try pdfplumber first; fall back to OCR if little/no text
    • Text files   → read directly

    Usage:
        loader = DocumentLoader()
        result = loader.load("report.pdf")
        print(result["text"][:200])
        print(result["metadata"])
    """

    def __init__(self, ocr_mode: OcrMode = "hybrid", ocr_min_confidence: float = 0.6):
        self.pdf_extractor = PDFExtractor()
        self.ocr_engine = OCREngine(mode=ocr_mode, min_confidence=ocr_min_confidence)
        self.ocr_mode = ocr_mode

    # Public API
    def load(self, file_path: str, force_ocr: bool = False, force_ai: bool = False, ocr_mode: Optional[OcrMode] = None) -> Dict[str, Any]:
        """Load a document and return its text + metadata."""
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Resolve effective OCR mode for this call
        effective_mode: OcrMode = ocr_mode if ocr_mode is not None else self.ocr_mode
        if force_ai:
            effective_mode = "mistral"

        suffix = path.suffix.lower()

        if suffix in PDF_EXTENSIONS:
            return self.load_pdf(file_path, force_ocr, effective_mode)
       
        elif suffix in IMAGE_EXTENSIONS:
            return self.load_image(file_path, effective_mode)
       
        elif suffix in TEXT_EXTENSIONS:
            return self.load_text(file_path)
       
        else:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: PDF, images ({', '.join(IMAGE_EXTENSIONS)}), "
                f"text ({', '.join(TEXT_EXTENSIONS)})"
            )

    def load_batch(self, file_paths: List[str]) -> str:
        """
        Batch OCR is not supported with the Azure curl-based endpoint.
        This method exists to keep the API surface but will always raise.
        """
        
        raise NotImplementedError(
            "Batch OCR is not supported with the Azure Mistral OCR endpoint. "
            "Call load() in a loop for multiple files."
        )

    def check_batch(self, job_id: str) -> Dict[str, Any]:
        """
        Placeholder for batch status API — not supported with Azure endpoint.
        """
        
        raise NotImplementedError("Batch status is not supported with the Azure OCR endpoint.")

    def get_batch(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Placeholder for batch result retrieval — not supported with Azure endpoint.
        """
        
        raise NotImplementedError("Batch result retrieval is not supported with the Azure OCR endpoint.")

    # Private Handlers
    def load_pdf(self, file_path: str, force_ocr: bool, ocr_mode: OcrMode) -> Dict[str, Any]:
        """Load a PDF — native extraction first, OCR as fallback."""
       
        metadata = self.pdf_extractor.get_metadata(file_path)

        needs_ocr = force_ocr or not self.pdf_extractor.has_extractable_text(file_path)

        if needs_ocr:
            logger.info(f"Using OCR for: {file_path}")
            ocr_result = self.ocr_engine.extract_from_pdf(file_path, force_ai=(ocr_mode == "mistral"), mode=ocr_mode)
            metadata["ocr_used"] = True
            metadata["ocr_confidence"] = ocr_result["confidence"]
            text = ocr_result["text"]
            mistral_ocr_response = ocr_result["mistral_ocr_response"]
            ai_extracted = ocr_result["ai_extracted"]
        
        else:
            logger.info(f"Using pdfplumber (native text) for: {file_path}")
            text = self.pdf_extractor.extract_text(file_path)
            metadata["ocr_used"] = False
            metadata["ocr_confidence"] = None
            mistral_ocr_response = None
            ai_extracted = False

        return {
            "text": text,
            "mistral_ocr_response": mistral_ocr_response,
            "metadata": metadata,
            "ai_extracted": ai_extracted,
        }

    def load_image(self, file_path: str, ocr_mode: OcrMode) -> Dict[str, Any]:
        """Load an image file via OCR."""
        
        logger.info(f"Using OCR for image: {file_path}")
        ocr_result = self.ocr_engine.extract_from_image(file_path, mode=ocr_mode)

        return {
            "text": ocr_result["text"],
            "mistral_ocr_response": ocr_result["mistral_ocr_response"],
            "metadata": {
                "file_path": str(Path(file_path).resolve()),
                "file_type": "image",
                "page_count": 1,
                "ocr_used": True,
                "ocr_confidence": ocr_result["confidence"],
            },
            "ai_extracted": ocr_result["ai_extracted"],
        }

    def load_text(self, file_path: str) -> Dict[str, Any]:
        """Load a plain text file directly."""
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return {
            "text": text,
            "metadata": {
                "file_path": str(Path(file_path).resolve()),
                "file_type": "text",
                "page_count": None,
                "ocr_used": False,
                "ocr_confidence": None,
            },
        }
