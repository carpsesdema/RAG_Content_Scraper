# utils/deduplicator.py

import hashlib
import difflib
from typing import List, Dict, Set, Tuple
import re
import ast


class SmartDeduplicator:
    """Advanced deduplication for code snippets."""

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.processed_snippets: List[Dict] = []

    def normalize_code(self, code: str) -> str:
        """Normalize code for better comparison."""
        # Remove comments and docstrings for comparison
        lines = []
        in_multiline_string = False
        quote_char = None

        for line in code.splitlines():
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Handle multiline strings/docstrings
            if '"""' in stripped or "'''" in stripped:
                if not in_multiline_string:
                    quote_char = '"""' if '"""' in stripped else "'''"
                    in_multiline_string = True
                    continue
                elif quote_char in stripped:
                    in_multiline_string = False
                    continue

            if in_multiline_string:
                continue

            # Remove single-line comments
            if stripped.startswith('#'):
                continue

            # Remove inline comments but keep the code
            if '#' in stripped:
                code_part = stripped.split('#')[0].strip()
                if code_part:
                    lines.append(code_part)
            else:
                lines.append(stripped)

        return '\n'.join(lines)

    def get_code_fingerprint(self, code: str) -> str:
        """Generate a fingerprint for code structure."""
        try:
            # Try to parse as Python AST for structural comparison
            tree = ast.parse(code)

            # Extract structural elements
            elements = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    elements.append(f"func:{node.name}")
                elif isinstance(node, ast.AsyncFunctionDef):
                    elements.append(f"asyncfunc:{node.name}")
                elif isinstance(node, ast.ClassDef):
                    elements.append(f"class:{node.name}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        elements.append(f"import:{alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    elements.append(f"from:{node.module}")

            return '|'.join(sorted(elements))

        except SyntaxError:
            # If not valid Python, use text-based fingerprinting
            normalized = self.normalize_code(code)
            # Extract key identifiers and keywords
            identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', normalized)
            return '|'.join(sorted(set(identifiers)))

    def calculate_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets."""
        # Normalize both codes
        norm1 = self.normalize_code(code1)
        norm2 = self.normalize_code(code2)

        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()

    def is_duplicate(self, code: str) -> Tuple[bool, str]:
        """Check if code is a duplicate or very similar to existing snippets."""

        # Quick hash check for exact duplicates
        code_hash = hashlib.md5(self.normalize_code(code).encode()).hexdigest()
        if code_hash in self.seen_hashes:
            return True, "exact_duplicate"

        # Check similarity with existing snippets
        for snippet_data in self.processed_snippets:
            similarity = self.calculate_similarity(code, snippet_data['code'])
            if similarity >= self.similarity_threshold:
                return True, f"similar_{similarity:.2f}"

        return False, "unique"

    def add_snippet(self, code: str, metadata: Dict = None) -> bool:
        """Add snippet if not duplicate. Returns True if added."""
        is_dup, reason = self.is_duplicate(code)

        if is_dup:
            return False

        # Add to our tracking
        code_hash = hashlib.md5(self.normalize_code(code).encode()).hexdigest()
        self.seen_hashes.add(code_hash)

        snippet_data = {
            'code': code,
            'hash': code_hash,
            'fingerprint': self.get_code_fingerprint(code),
            'metadata': metadata or {}
        }
        self.processed_snippets.append(snippet_data)

        return True

    def deduplicate_list(self, snippets: List[str],
                         metadatas: List[Dict] = None) -> List[Dict]:
        """Deduplicate a list of snippets."""
        if metadatas is None:
            metadatas = [{}] * len(snippets)

        deduped = []

        for snippet, metadata in zip(snippets, metadatas):
            if self.add_snippet(snippet, metadata):
                deduped.append({
                    'code': snippet,
                    'metadata': metadata
                })

        return deduped

    def get_stats(self) -> Dict:
        """Get deduplication statistics."""
        return {
            'total_processed': len(self.processed_snippets),
            'unique_hashes': len(self.seen_hashes),
            'similarity_threshold': self.similarity_threshold
        }


class SemanticDeduplicator:
    """More advanced semantic deduplication (requires additional libraries)."""

    def __init__(self):
        # This would use sentence-transformers or similar for semantic similarity
        # For now, placeholder implementation
        pass

    def semantic_similarity(self, code1: str, code2: str) -> float:
        """Calculate semantic similarity (placeholder)."""
        # Would use embeddings here in a full implementation
        # For now, fall back to text similarity
        matcher = difflib.SequenceMatcher(None, code1, code2)
        return matcher.ratio()