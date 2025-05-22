# gui/main_window.py - Enhanced with RAG Export Features and Freelancer Tools

import sys
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QPlainTextEdit, QFileDialog,
    QLabel, QComboBox, QProgressBar, QCheckBox,
    QGroupBox, QTextEdit, QTabWidget, QSplitter
)
from scraper.searcher import search_and_fetch
from storage.saver import save_snippets
from utils.logger import setup_logger

try:
    from config import (
        DEFAULT_WINDOW_TITLE, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT,
        STYLESHEET_PATH, SEARCH_SOURCES_COUNT, FREELANCE_MODE,
        EMBEDDING_RAG_EXPORT_ENABLED, DUAL_LLM_EXPORT, CODE_CATEGORIZATION_ENABLED
    )
except ImportError:
    # Fallback defaults if config.py is not found
    print("Warning: config.py not found or not accessible, using fallback GUI settings.", file=sys.stderr)
    DEFAULT_WINDOW_TITLE = "RAG Content Scraper (Fallback)"
    DEFAULT_WINDOW_WIDTH = 850
    DEFAULT_WINDOW_HEIGHT = 650
    STYLESHEET_PATH = ""
    SEARCH_SOURCES_COUNT = 4
    FREELANCE_MODE = False
    EMBEDDING_RAG_EXPORT_ENABLED = False
    DUAL_LLM_EXPORT = False
    CODE_CATEGORIZATION_ENABLED = False


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
                from scraper.fetcher import fetch_url
                from scraper.parser import extract_code
                html = fetch_url(self.query)
                self.progress.emit(50, "Extracting code from URL...")
                snippets = extract_code(html)
                self.progress.emit(100, "URL processing complete.")
            else:
                # Progress callback for multi-source search
                def progress_callback(message, percentage_step):
                    if self.total_sources > 0:
                        current_progress = int(20 + (self.current_source_index / self.total_sources) * 70)
                    else:
                        current_progress = 20
                    self.progress.emit(current_progress, message)
                    self.current_source_index += 1

                self.progress.emit(20, f"Starting enhanced search for: {self.query}...")
                snippets = search_and_fetch(self.query, self.logger, progress_callback)

            self.progress.emit(95, "Processing results...")
            unique_snippets = list(dict.fromkeys(snippets))
            self.progress.emit(100, "Search complete.")
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
            save_snippets(self.snippets, self.directory)
            self.finished.emit(f"Saved {len(self.snippets)} snippets to {self.directory}.")
        except Exception as e:
            self.logger.exception("Error saving snippets in SaveWorker")
            user_friendly_message = f"Error Type: {type(e).__name__}\nMessage: {str(e)}"
            self.error.emit(user_friendly_message)


class EnhancedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = setup_logger()
        self.setWindowTitle(DEFAULT_WINDOW_TITLE)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.enhanced_snippet_data = []  # Store enhanced data for RAG export
        self.categorization_results = {}  # Store categorization results
        self._setup_enhanced_ui()
        self.snippets = []

    def _setup_enhanced_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # === TOP SECTION: Input and Controls ===
        input_group = QGroupBox("Search Configuration")
        input_layout = QVBoxLayout(input_group)

        # Mode and query row
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Search", "URL"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter module name or search query")

        self.fetch_button = QPushButton("🔍 Fetch Content")
        self.fetch_button.clicked.connect(self.on_fetch)

        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(QLabel("Query/URL:"))
        mode_layout.addWidget(self.url_input, 1)
        mode_layout.addWidget(self.fetch_button)
        input_layout.addLayout(mode_layout)

        # Enhanced options (if freelance mode enabled)
        if FREELANCE_MODE:
            options_layout = QHBoxLayout()

            self.freelance_mode_cb = QCheckBox("Freelance Focus")
            self.freelance_mode_cb.setChecked(True)
            self.freelance_mode_cb.setToolTip("Prioritize freelance-relevant code examples")

            self.high_value_only_cb = QCheckBox("High-Value Only")
            self.high_value_only_cb.setToolTip("Show only high client-value code snippets")

            self.include_patterns_cb = QCheckBox("Include Patterns")
            self.include_patterns_cb.setChecked(True)
            self.include_patterns_cb.setToolTip("Include design patterns and best practices")

            options_layout.addWidget(self.freelance_mode_cb)
            options_layout.addWidget(self.high_value_only_cb)
            options_layout.addWidget(self.include_patterns_cb)
            options_layout.addStretch()

            input_layout.addLayout(options_layout)

        main_layout.addWidget(input_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # === MIDDLE SECTION: Results Display ===
        results_splitter = QSplitter(Qt.Horizontal)

        # Left side: Snippet content
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.results_tabs = QTabWidget()

        # Raw snippets tab
        self.snippets_edit = QPlainTextEdit()
        self.snippets_edit.setReadOnly(True)
        self.snippets_edit.setPlaceholderText("Fetched code snippets will appear here...")
        self.results_tabs.addTab(self.snippets_edit, "📄 Code Snippets")

        # Analysis tab (if categorization enabled)
        if CODE_CATEGORIZATION_ENABLED:
            self.analysis_edit = QTextEdit()
            self.analysis_edit.setReadOnly(True)
            self.analysis_edit.setPlaceholderText("Code analysis and categorization results will appear here...")
            self.results_tabs.addTab(self.analysis_edit, "📊 Analysis")

        left_layout.addWidget(self.results_tabs)
        results_splitter.addWidget(left_widget)

        # Right side: Metadata and insights (if enhanced features available)
        if FREELANCE_MODE or CODE_CATEGORIZATION_ENABLED:
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)

            # Insights panel
            insights_group = QGroupBox("💡 Insights")
            insights_layout = QVBoxLayout(insights_group)

            self.insights_text = QTextEdit()
            self.insights_text.setReadOnly(True)
            self.insights_text.setMaximumHeight(200)
            self.insights_text.setPlaceholderText("Freelance insights and recommendations will appear here...")
            insights_layout.addWidget(self.insights_text)

            right_layout.addWidget(insights_group)

            # Categories panel
            if CODE_CATEGORIZATION_ENABLED:
                categories_group = QGroupBox("🏷️ Categories")
                categories_layout = QVBoxLayout(categories_group)

                self.categories_text = QTextEdit()
                self.categories_text.setReadOnly(True)
                self.categories_text.setMaximumHeight(150)
                self.categories_text.setPlaceholderText("Code categories will appear here...")
                categories_layout.addWidget(self.categories_text)

                right_layout.addWidget(categories_group)

            # Query suggestions
            suggestions_group = QGroupBox("🔍 Related Queries")
            suggestions_layout = QVBoxLayout(suggestions_group)

            self.suggestions_text = QTextEdit()
            self.suggestions_text.setReadOnly(True)
            self.suggestions_text.setMaximumHeight(100)
            self.suggestions_text.setPlaceholderText("Related search suggestions will appear here...")
            suggestions_layout.addWidget(self.suggestions_text)

            right_layout.addWidget(suggestions_group)
            right_layout.addStretch()

            results_splitter.addWidget(right_widget)
            results_splitter.setSizes([600, 300])  # Give more space to code snippets

        main_layout.addWidget(results_splitter)

        # === BOTTOM SECTION: Export and Actions ===
        bottom_group = QGroupBox("Export Options")
        bottom_layout = QVBoxLayout(bottom_group)

        # Standard export row
        standard_layout = QHBoxLayout()
        self.save_button = QPushButton("💾 Save All Snippets")
        self.save_button.clicked.connect(self.on_save)
        self.save_button.setEnabled(False)

        standard_layout.addWidget(self.save_button)
        standard_layout.addStretch()
        bottom_layout.addLayout(standard_layout)

        # Enhanced export row (if available)
        if EMBEDDING_RAG_EXPORT_ENABLED:
            enhanced_layout = QHBoxLayout()

            self.export_format_combo = QComboBox()
            self.export_format_combo.addItems(["Standard", "RAG-JSONL", "RAG-Markdown", "RAG-XML", "RAG-YAML"])

            self.rag_export_button = QPushButton("🤖 Export for RAG")
            self.rag_export_button.clicked.connect(self.on_rag_export)
            self.rag_export_button.setEnabled(False)

            if DUAL_LLM_EXPORT:
                self.dual_llm_export_button = QPushButton("🔄 Dual LLM Export")
                self.dual_llm_export_button.clicked.connect(self.on_dual_llm_export)
                self.dual_llm_export_button.setEnabled(False)
                enhanced_layout.addWidget(self.dual_llm_export_button)

            enhanced_layout.addWidget(QLabel("Format:"))
            enhanced_layout.addWidget(self.export_format_combo)
            enhanced_layout.addWidget(self.rag_export_button)
            enhanced_layout.addStretch()

            bottom_layout.addLayout(enhanced_layout)

        # Status row
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready for enhanced RAG content scraping.")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        status_layout.addWidget(self.status_label, 1)
        bottom_layout.addLayout(status_layout)

        main_layout.addWidget(bottom_group)

    def on_mode_change(self, mode):
        if mode == "URL":
            self.url_input.setPlaceholderText("Enter URL here (e.g., https://...)")
        else:
            if FREELANCE_MODE:
                self.url_input.setPlaceholderText("Enter query (e.g., fastapi authentication, stripe integration)")
            else:
                self.url_input.setPlaceholderText(
                    "Enter module name or search query (e.g., asyncio, pandas http client)")

    def on_fetch(self):
        query = self.url_input.text().strip()
        mode = self.mode_combo.currentText()

        if not query:
            self.status_label.setText("Please enter a query or URL.")
            return

        # Clear previous results
        self.enhanced_snippet_data = []
        self.categorization_results = {}

        self.fetch_button.setEnabled(False)
        self.save_button.setEnabled(False)
        if hasattr(self, 'rag_export_button'):
            self.rag_export_button.setEnabled(False)
        if hasattr(self, 'dual_llm_export_button'):
            self.dual_llm_export_button.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Initializing enhanced {mode} for: {query}...")
        self.snippets_edit.setPlainText("")

        if hasattr(self, 'analysis_edit'):
            self.analysis_edit.setHtml("")
        if hasattr(self, 'insights_text'):
            self.insights_text.setHtml("")
        if hasattr(self, 'categories_text'):
            self.categories_text.setHtml("")
        if hasattr(self, 'suggestions_text'):
            self.suggestions_text.setHtml("")

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
            self._display_enhanced_results()

        # Display basic snippets
        display_text = "\n\n# -----\n\n".join(self.snippets) if self.snippets else "No code snippets found."
        self.snippets_edit.setPlainText(display_text)

        self.status_label.setText(status_message)
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        if self.snippets:
            self.save_button.setEnabled(True)
            if hasattr(self, 'rag_export_button'):
                self.rag_export_button.setEnabled(True)
            if hasattr(self, 'dual_llm_export_button'):
                self.dual_llm_export_button.setEnabled(True)

    def _display_enhanced_results(self):
        """Display enhanced analysis results."""
        if not self.enhanced_snippet_data:
            return

        # Generate analysis display
        if hasattr(self, 'analysis_edit'):
            analysis_html = self._generate_analysis_html()
            self.analysis_edit.setHtml(analysis_html)

        # Generate insights
        if hasattr(self, 'insights_text'):
            insights_html = self._generate_insights_html()
            self.insights_text.setHtml(insights_html)

        # Generate categories
        if hasattr(self, 'categories_text'):
            categories_html = self._generate_categories_html()
            self.categories_text.setHtml(categories_html)

        # Generate suggestions
        if hasattr(self, 'suggestions_text'):
            suggestions_html = self._generate_suggestions_html()
            self.suggestions_text.setHtml(suggestions_html)

    def _generate_analysis_html(self):
        """Generate HTML for analysis tab."""
        if not self.enhanced_snippet_data:
            return "<p>No analysis data available.</p>"

        html = "<h3>📊 Code Analysis Summary</h3>"

        # Overall statistics
        total_snippets = len(self.enhanced_snippet_data)
        avg_score = sum(item.get('score', 0) for item in self.enhanced_snippet_data) / total_snippets
        freelance_relevant = sum(1 for item in self.enhanced_snippet_data
                                 if item['metadata'].get('freelance_relevant', False))

        html += f"""
        <p><strong>Total Snippets:</strong> {total_snippets}</p>
        <p><strong>Average Quality Score:</strong> {avg_score:.1f}</p>
        <p><strong>Freelance Relevant:</strong> {freelance_relevant} ({freelance_relevant / total_snippets * 100:.1f}%)</p>
        """

        # Top snippets by score
        sorted_snippets = sorted(self.enhanced_snippet_data, key=lambda x: x.get('score', 0), reverse=True)
        html += "<h4>🏆 Top Quality Snippets</h4><ul>"

        for i, snippet in enumerate(sorted_snippets[:5]):
            metadata = snippet['metadata']
            score = snippet.get('score', 0)
            complexity = metadata.get('complexity', 'unknown')
            categories = ', '.join(metadata.get('categories', [])[:3])

            html += f"""
            <li><strong>Score: {score:.1f}</strong> - {complexity} complexity
            <br>Categories: {categories}
            <br>Client Value: {metadata.get('client_value', 'unknown')}</li>
            """

        html += "</ul>"
        return html

    def _generate_insights_html(self):
        """Generate HTML for freelance insights."""
        if not self.enhanced_snippet_data:
            return "<p>No insights available.</p>"

        html = "<h4>💰 Freelance Value Analysis</h4>"

        # High-value snippets
        high_value_snippets = [item for item in self.enhanced_snippet_data
                               if 'high_value' in item['metadata'].get('client_value', '')]

        if high_value_snippets:
            html += f"<p><strong>{len(high_value_snippets)} high-value snippets found!</strong></p>"
            html += "<ul>"
            for snippet in high_value_snippets[:3]:
                client_value = snippet['metadata'].get('client_value', '')
                use_cases = snippet['metadata'].get('use_cases', [])
                html += f"<li>{client_value.replace('_', ' ').title()}"
                if use_cases:
                    html += f" - {', '.join(use_cases[:2])}"
                html += "</li>"
            html += "</ul>"

        # Common patterns for freelance work
        all_patterns = []
        for item in self.enhanced_snippet_data:
            all_patterns.extend(item['metadata'].get('patterns', []))

        if all_patterns:
            from collections import Counter
            common_patterns = Counter(all_patterns).most_common(5)
            html += "<h5>🔧 Common Patterns</h5><ul>"
            for pattern, count in common_patterns:
                html += f"<li>{pattern.replace('_', ' ').title()}: {count}</li>"
            html += "</ul>"

        return html

    def _generate_categories_html(self):
        """Generate HTML for categories display."""
        if not self.enhanced_snippet_data:
            return "<p>No categories available.</p>"

        # Collect all categories
        all_categories = []
        for item in self.enhanced_snippet_data:
            all_categories.extend(item['metadata'].get('categories', []))

        if not all_categories:
            return "<p>No categories detected.</p>"

        from collections import Counter
        category_counts = Counter(all_categories)

        html = "<h4>📂 Category Distribution</h4><ul>"
        for category, count in category_counts.most_common(10):
            html += f"<li><strong>{category.replace('_', ' ').title()}:</strong> {count}</li>"
        html += "</ul>"

        return html

    def _generate_suggestions_html(self):
        """Generate HTML for query suggestions."""
        # Try to get suggestions from categorizer
        try:
            from utils.code_categorizer import CodeCategorizer
            categorizer = CodeCategorizer()

            if self.enhanced_snippet_data:
                # Use first snippet's categorization for suggestions
                first_snippet_meta = self.enhanced_snippet_data[0]['metadata']
                suggestions = categorizer.suggest_related_queries(first_snippet_meta)

                html = "<h4>🔍 Related Searches</h4><ul>"
                for suggestion in suggestions[:8]:
                    html += f"<li>{suggestion}</li>"
                html += "</ul>"
                return html
        except ImportError:
            pass

        return "<p>No suggestions available.</p>"

    def handle_fetch_error(self, error_message):
        self.logger.error(f"GUI received fetch error summary: {error_message}")
        self.snippets_edit.setPlainText(f"An error occurred during fetch:\n\n{error_message}")
        self.status_label.setText("Error during fetch operation. See log for details.")
        self.fetch_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_button.setEnabled(False)
        if hasattr(self, 'rag_export_button'):
            self.rag_export_button.setEnabled(False)
        if hasattr(self, 'dual_llm_export_button'):
            self.dual_llm_export_button.setEnabled(False)

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
        if hasattr(self, 'rag_export_button'):
            self.rag_export_button.setEnabled(False)
        if hasattr(self, 'dual_llm_export_button'):
            self.dual_llm_export_button.setEnabled(False)

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
            if hasattr(self, 'rag_export_button'):
                self.rag_export_button.setEnabled(True)
            if hasattr(self, 'dual_llm_export_button'):
                self.dual_llm_export_button.setEnabled(True)
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
            if hasattr(self, 'rag_export_button'):
                self.rag_export_button.setEnabled(True)
            if hasattr(self, 'dual_llm_export_button'):
                self.dual_llm_export_button.setEnabled(True)
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
            from utils.code_categorizer import CodeCategorizer

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
                categorizer = CodeCategorizer()
                snippets_data = []
                for snippet in self.snippets:
                    categorization = categorizer.categorize_snippet(snippet)
                    snippets_data.append({
                        'code': snippet,
                        'score': categorization.get('freelance_score', 0) + 5,
                        'metadata': categorization
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

    def on_dual_llm_export(self):
        """Handle dual LLM export with multiple optimized formats."""
        if not self.snippets:
            self.status_label.setText("No snippets to export.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Dual LLM Export")
        if not directory:
            self.status_label.setText("Export cancelled.")
            return

        try:
            from storage.embedding_rag_exporter import EmbeddingRAGExporter
            from utils.code_categorizer import CodeCategorizer

            # Enhanced categorization
            categorizer = CodeCategorizer()
            enhanced_data = []

            if hasattr(self, 'enhanced_snippet_data') and self.enhanced_snippet_data:
                # Use existing enhanced data
                enhanced_data = self.enhanced_snippet_data
            else:
                # Create enhanced categorization
                for snippet in self.snippets:
                    categorization = categorizer.categorize_snippet(snippet)
                    enhanced_data.append({
                        'code': snippet,
                        'score': categorization.get('freelance_score', 0) + 5,
                        'metadata': categorization
                    })

            exporter = EmbeddingRAGExporter()
            query = self.url_input.text().strip() or "search_query"

            export_files = exporter.export_for_dual_llm(enhanced_data, directory, query)

            files_created = len(export_files)
            file_types = ', '.join(export_files.keys())
            self.status_label.setText(f"Dual LLM export complete: {files_created} files created ({file_types})")

        except ImportError:
            self.status_label.setText("Dual LLM export features not available.")
        except Exception as e:
            self.logger.error(f"Dual LLM export error: {e}")
            self.status_label.setText(f"Export failed: {str(e)}")

    def closeEvent(self, event):
        if hasattr(self, 'fetch_worker') and self.fetch_worker.isRunning():
            self.logger.info("Attempting to quit fetch_worker...")
            self.fetch_worker.quit()
            if not self.fetch_worker.wait(3000):  # Wait 3 seconds
                self.logger.warning("Fetch worker did not terminate gracefully, forcing termination.")
                self.fetch_worker.terminate()
                self.fetch_worker.wait()
        if hasattr(self, 'save_worker') and self.save_worker.isRunning():
            self.logger.info("Attempting to quit save_worker...")
            self.save_worker.quit()
            if not self.save_worker.wait(3000):  # Wait 3 seconds
                self.logger.warning("Save worker did not terminate gracefully, forcing termination.")
                self.save_worker.terminate()
                self.save_worker.wait()
        event.accept()


# Create alias for backward compatibility
MainWindow = EnhancedMainWindow

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

    window = EnhancedMainWindow()
    window.show()
    sys.exit(app.exec())