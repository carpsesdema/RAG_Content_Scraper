# gui/main_window.py - Enhanced with RAG Export Features

import sys
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QPlainTextEdit, QFileDialog,
    QLabel, QComboBox, QProgressBar
)
from scraper.searcher import search_and_fetch
from storage.saver import save_snippets
from utils.logger import setup_logger

try:
    from config import (
        DEFAULT_WINDOW_TITLE, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT,
        STYLESHEET_PATH, SEARCH_SOURCES_COUNT
    )
except ImportError:
    # Fallback defaults if config.py is not found
    print("Warning: config.py not found or not accessible, using fallback GUI settings.", file=sys.stderr)
    DEFAULT_WINDOW_TITLE = "RAG Content Scraper (Fallback)"
    DEFAULT_WINDOW_WIDTH = 850
    DEFAULT_WINDOW_HEIGHT = 650
    STYLESHEET_PATH = ""  # No stylesheet if config is missing
    SEARCH_SOURCES_COUNT = 4  # Default if not in config


class FetchWorker(QThread):
    progress = Signal(int, str)  # value, message
    finished = Signal(list, str)  # snippets, status_message
    error = Signal(str)  # error_message

    def __init__(self, query, mode, logger):
        super().__init__()
        self.query = query
        self.mode = mode
        self.logger = logger
        self.current_source_index = 0
        # In URL mode, it's 1 direct step.
        # In Search mode, total_sources is based on the number of search functions in search_and_fetch
        self.total_sources = 1 if self.mode == "URL" else SEARCH_SOURCES_COUNT

    def run(self):
        try:
            snippets = []
            if self.mode == "URL":
                self.progress.emit(25, f"Fetching URL: {self.query}...")
                from scraper.fetcher import fetch_url  # Keep local import for clarity
                from scraper.parser import extract_code  # Keep local import for clarity
                html = fetch_url(self.query)  # Uses configured timeout via fetcher
                self.progress.emit(50, "Extracting code from URL...")
                snippets = extract_code(html)
                self.progress.emit(100, "URL processing complete.")
            else:
                # Progress callback for multi-source search
                def progress_callback(message, percentage_step):
                    # current_progress will be driven by search_and_fetch's own steps
                    # but we can map it based on source index if needed, or just pass its message
                    # The percentage_step from search_and_fetch might be more accurate
                    # For now, let's use a simpler approach based on source index
                    # Max 75-80% for fetching, rest for deduplication.
                    # The SEARCH_SOURCES_COUNT is the number of main fetching steps.
                    # Initial 20% is for "Starting search..."
                    # Then distribute (90-20)=70% over self.total_sources
                    if self.total_sources > 0:
                        current_progress = int(20 + (self.current_source_index / self.total_sources) * 70)
                    else:  # Should not happen in Search mode
                        current_progress = 20
                    self.progress.emit(current_progress, message)
                    self.current_source_index += 1  # Increment when a source is processed

                self.progress.emit(20, f"Starting search for: {self.query}...")
                # Pass the progress_callback to search_and_fetch
                # search_and_fetch will call it after each source.
                snippets = search_and_fetch(self.query, self.logger, progress_callback)

            self.progress.emit(95, "Deduplicating snippets...")  # Increased from 90
            unique_snippets = list(dict.fromkeys(snippets))
            self.progress.emit(100, "Fetch complete.")
            self.finished.emit(unique_snippets, f"Found {len(unique_snippets)} unique snippets.")

        except Exception as e:
            self.logger.exception(f"Error in FetchWorker for query '{self.query}' (mode: {self.mode})")
            user_friendly_message = f"Error Type: {type(e).__name__}\nMessage: {str(e)}"
            self.error.emit(user_friendly_message)


class SaveWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, snippets, directory, logger):
        super().__init__()
        self.snippets = snippets
        self.directory = directory
        self.logger = logger

    def run(self):
        try:
            save_snippets(self.snippets, self.directory)  # Uses configured slug length via saver
            self.finished.emit(f"Saved {len(self.snippets)} snippets to {self.directory}.")
        except Exception as e:
            self.logger.exception("Error saving snippets in SaveWorker")
            user_friendly_message = f"Error Type: {type(e).__name__}\nMessage: {str(e)}"
            self.error.emit(user_friendly_message)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = setup_logger()  # Uses configured logger settings
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.enhanced_snippet_data = []  # Store enhanced data for RAG export
        self._setup_ui()
        self.snippets = []

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        input_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Search", "URL"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter module name or search query")

        self.fetch_button = QPushButton("Fetch Content")
        self.fetch_button.clicked.connect(self.on_fetch)

        input_layout.addWidget(QLabel("Mode:"))
        input_layout.addWidget(self.mode_combo)
        input_layout.addWidget(QLabel("Query/URL:"))
        input_layout.addWidget(self.url_input, 1)
        input_layout.addWidget(self.fetch_button)
        layout.addLayout(input_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.snippets_edit = QPlainTextEdit()
        self.snippets_edit.setReadOnly(True)
        self.snippets_edit.setPlaceholderText("Fetched code snippets will appear here...")
        layout.addWidget(self.snippets_edit)

        bottom_layout = QHBoxLayout()
        self.save_button = QPushButton("Save All Snippets")
        self.save_button.clicked.connect(self.on_save)
        self.save_button.setEnabled(False)

        # Enhanced export options
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["Standard", "RAG-JSONL", "RAG-Markdown", "RAG-XML", "RAG-YAML"])

        self.rag_export_button = QPushButton("Export for RAG")
        self.rag_export_button.clicked.connect(self.on_rag_export)
        self.rag_export_button.setEnabled(False)

        self.status_label = QLabel("Ready.")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        bottom_layout.addWidget(self.save_button)
        bottom_layout.addWidget(QLabel("Format:"))
        bottom_layout.addWidget(self.export_format_combo)
        bottom_layout.addWidget(self.rag_export_button)
        bottom_layout.addWidget(self.status_label, 1)
        layout.addLayout(bottom_layout)

    def on_mode_change(self, mode):
        if mode == "URL":
            self.url_input.setPlaceholderText("Enter URL here (e.g., https://...)")
        else:
            self.url_input.setPlaceholderText("Enter module name or search query (e.g., asyncio, pandas http client)")

    def on_fetch(self):
        query = self.url_input.text().strip()
        mode = self.mode_combo.currentText()

        if not query:
            self.status_label.setText("Please enter a query or URL.")
            return

        self.fetch_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.rag_export_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Initializing {mode} for: {query}...")
        self.snippets_edit.setPlainText("")

        self.fetch_worker = FetchWorker(query, mode, self.logger)
        self.fetch_worker.progress.connect(self.update_fetch_progress)
        self.fetch_worker.finished.connect(self.handle_fetch_finished)
        self.fetch_worker.error.connect(self.handle_fetch_error)
        self.fetch_worker.start()

    def update_fetch_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def handle_fetch_finished(self, snippets, status_message):
        self.snippets = snippets

        # Try to get enhanced data from logger
        if hasattr(self.logger, 'enhanced_snippet_data'):
            self.enhanced_snippet_data = self.logger.enhanced_snippet_data

        display_text = "\n\n# -----\n\n".join(self.snippets) if self.snippets else "No code snippets found."
        self.snippets_edit.setPlainText(display_text)
        self.status_label.setText(status_message)
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        if self.snippets:
            self.save_button.setEnabled(True)
            self.rag_export_button.setEnabled(True)

    def handle_fetch_error(self, error_message):
        self.logger.error(f"GUI received fetch error summary: {error_message}")
        self.snippets_edit.setPlainText(f"An error occurred during fetch:\n\n{error_message}")
        self.status_label.setText("Error during fetch operation. See log for details.")  # Minor change for consistency
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_button.setEnabled(False)
        self.rag_export_button.setEnabled(False)

    def on_save(self):
        if not self.snippets:
            self.status_label.setText("No snippets to save.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Snippets")
        if not directory:
            self.status_label.setText("Save cancelled.")
            return

        self.save_button.setEnabled(False)
        self.fetch_button.setEnabled(False)
        self.rag_export_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText(f"Saving {len(self.snippets)} snippets...")

        self.save_worker = SaveWorker(self.snippets, directory, self.logger)
        self.save_worker.finished.connect(self.handle_save_finished)
        self.save_worker.error.connect(self.handle_save_error)
        self.save_worker.start()

    def handle_save_finished(self, status_message):
        self.status_label.setText(status_message)
        self.save_button.setEnabled(True)
        self.fetch_button.setEnabled(True)
        if self.snippets:
            self.rag_export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)  # Reset for fetch

    def handle_save_error(self, error_message):
        self.logger.error(f"GUI received save error summary: {error_message}")
        error_type_for_status = "Unknown Error"
        try:
            first_line = error_message.split('\n', 1)[0]
            if first_line.startswith("Error Type: "):
                error_type_for_status = first_line.replace("Error Type: ", "").strip()
        except Exception:
            pass
        self.status_label.setText(f"Save Error ({error_type_for_status}). See log for details.")
        self.save_button.setEnabled(True)
        self.fetch_button.setEnabled(True)
        if self.snippets:
            self.rag_export_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)

    def on_rag_export(self):
        """Handle RAG export with enhanced formats."""
        if not self.snippets:
            self.status_label.setText("No snippets to export.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory for RAG Export")
        if not directory:
            self.status_label.setText("Export cancelled.")
            return

        try:
            from storage.rag_exporter import RAGExporter
            from scraper.quality_filter import CodeQualityFilter

            format_map = {
                "RAG-JSONL": "jsonl",
                "RAG-Markdown": "markdown",
                "RAG-XML": "xml",
                "RAG-YAML": "yaml"
            }

            selected_format = self.export_format_combo.currentText()

            if selected_format == "Standard":
                # Use original save method
                self.on_save()
                return

            # Use enhanced data if available, otherwise create it
            if hasattr(self, 'enhanced_snippet_data') and self.enhanced_snippet_data:
                snippets_data = self.enhanced_snippet_data
            else:
                # Create enhanced data from basic snippets
                quality_filter = CodeQualityFilter()
                snippets_data = []
                for snippet in self.snippets:
                    result = quality_filter.score_snippet(snippet)
                    snippets_data.append({
                        'code': snippet,
                        'score': result['score'],
                        'metadata': result['metadata']
                    })

            exporter = RAGExporter()
            query = self.url_input.text().strip() or "search_query"

            output_file = exporter.export_for_rag(
                snippets_data,
                directory,
                query,
                format_map[selected_format]
            )

            self.status_label.setText(f"RAG export complete: {output_file}")

        except ImportError:
            self.status_label.setText("Enhanced export features not available.")
        except Exception as e:
            self.logger.error(f"RAG export error: {e}")
            self.status_label.setText(f"RAG export failed: {str(e)}")

    def closeEvent(self, event):
        if hasattr(self, 'fetch_worker') and self.fetch_worker.isRunning():
            self.logger.info("Attempting to quit fetch_worker...")
            self.fetch_worker.quit()
            if not self.fetch_worker.wait(3000):  # Wait 3 seconds
                self.logger.warning("Fetch worker did not terminate gracefully, forcing termination.")
                self.fetch_worker.terminate()  # Force if not quit
                self.fetch_worker.wait()  # Wait again after terminate
        if hasattr(self, 'save_worker') and self.save_worker.isRunning():
            self.logger.info("Attempting to quit save_worker...")
            self.save_worker.quit()
            if not self.save_worker.wait(3000):  # Wait 3 seconds
                self.logger.warning("Save worker did not terminate gracefully, forcing termination.")
                self.save_worker.terminate()
                self.save_worker.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Attempt to load stylesheet from configured path
    actual_stylesheet_path = STYLESHEET_PATH
    if actual_stylesheet_path:  # Check if path is not empty
        try:
            with open(actual_stylesheet_path, encoding="utf-8") as f:
                app.setStyleSheet(f.read())
            print(f"Stylesheet '{actual_stylesheet_path}' loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: Stylesheet not found at '{actual_stylesheet_path}'. Using default styles.")
        except Exception as e:
            print(f"Could not load stylesheet from '{actual_stylesheet_path}': {e}")
    else:
        print("No stylesheet path configured. Using default styles.")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())