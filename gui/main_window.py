# gui/main_window.py - Enhanced with Pinescript Support

import sys
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QPlainTextEdit, QFileDialog,
    QLabel, QComboBox, QProgressBar, QCheckBox,
    QGroupBox, QTextEdit, QTabWidget, QSplitter,
    QButtonGroup, QRadioButton
)
from scraper.searcher import search_and_fetch, detect_content_type
from storage.saver import save_snippets
from utils.logger import setup_logger

try:
    from config import (
        DEFAULT_WINDOW_TITLE, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT,
        STYLESHEET_PATH, SEARCH_SOURCES_COUNT, FREELANCE_MODE,
        EMBEDDING_RAG_EXPORT_ENABLED, DUAL_LLM_EXPORT, CODE_CATEGORIZATION_ENABLED,
        # NEW: Pinescript settings
        PINESCRIPT_ENABLED, CONTENT_TYPES, DEFAULT_CONTENT_TYPE,
        LANGUAGE_DISPLAY_NAMES, SHOW_LANGUAGE_SELECTOR,
        TRADING_MODE, LANGUAGE_SPECIFIC_UI_HINTS
    )
except ImportError:
    # Fallback defaults
    print("Warning: config.py not found or not accessible, using fallback GUI settings.", file=sys.stderr)
    DEFAULT_WINDOW_TITLE = "RAG Content Scraper (Enhanced)"
    DEFAULT_WINDOW_WIDTH = 950  # Slightly wider for new UI elements
    DEFAULT_WINDOW_HEIGHT = 750
    STYLESHEET_PATH = ""
    SEARCH_SOURCES_COUNT = 4
    FREELANCE_MODE = False
    EMBEDDING_RAG_EXPORT_ENABLED = False
    DUAL_LLM_EXPORT = False
    CODE_CATEGORIZATION_ENABLED = False
    PINESCRIPT_ENABLED = False
    CONTENT_TYPES = {'python': True, 'pinescript': False}
    DEFAULT_CONTENT_TYPE = 'python'
    LANGUAGE_DISPLAY_NAMES = {'python': 'Python', 'pinescript': 'Pinescript', 'auto': 'Auto-Detect'}
    SHOW_LANGUAGE_SELECTOR = True
    TRADING_MODE = False
    LANGUAGE_SPECIFIC_UI_HINTS = True


class FetchWorker(QThread):
    progress = Signal(int, str)  # value, message
    finished = Signal(list, str)  # snippets, status_message
    error = Signal(str)  # error_message

    def __init__(self, query, mode, content_type, logger):
        super().__init__()
        self.query = query
        self.mode = mode
        self.content_type = content_type  # NEW
        self.logger = logger
        self.current_source_index = 0

        # Adjust total sources based on content type
        if content_type == 'pinescript':
            self.total_sources = 1  # Pinescript sources
        else:
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

                self.progress.emit(20, f"Starting enhanced search for: {self.query} ({self.content_type})...")
                # Pass content_type to the searcher
                snippets = search_and_fetch(self.query, self.logger, progress_callback, self.content_type)

            self.progress.emit(95, "Processing results...")
            unique_snippets = list(dict.fromkeys(snippets))
            self.progress.emit(100, "Search complete.")
            self.finished.emit(unique_snippets, f"Found {len(unique_snippets)} unique {self.content_type} snippets.")

        except Exception as e:
            self.logger.exception(
                f"Error in FetchWorker for query '{self.query}' (mode: {self.mode}, type: {self.content_type})")
            user_friendly_message = f"Error Type: {type(e).__name__}\nMessage: {str(e)}"
            self.error.emit(user_friendly_message)


class SaveWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, snippets, directory, content_type, logger):
        super().__init__()
        self.snippets = snippets
        self.directory = directory
        self.content_type = content_type  # NEW
        self.logger = logger

    def run(self):
        try:
            save_snippets(self.snippets, self.directory)
            self.finished.emit(f"Saved {len(self.snippets)} {self.content_type} snippets to {self.directory}.")
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
        self.enhanced_snippet_data = []
        self.categorization_results = {}
        self.current_content_type = DEFAULT_CONTENT_TYPE  # NEW: Track current content type
        self._setup_enhanced_ui()
        self.snippets = []

    def _setup_enhanced_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # === TOP SECTION: Input and Controls ===
        input_group = QGroupBox("Search Configuration")
        input_layout = QVBoxLayout(input_group)

        # NEW: Content Type Selection Row
        if SHOW_LANGUAGE_SELECTOR and (PINESCRIPT_ENABLED or len(CONTENT_TYPES) > 1):
            content_type_layout = QHBoxLayout()
            content_type_layout.addWidget(QLabel("Content Type:"))

            self.content_type_combo = QComboBox()

            # Add available content types
            if CONTENT_TYPES.get('python', True):
                self.content_type_combo.addItem(LANGUAGE_DISPLAY_NAMES.get('python', 'Python'), 'python')
            if CONTENT_TYPES.get('pinescript', False):
                self.content_type_combo.addItem(LANGUAGE_DISPLAY_NAMES.get('pinescript', 'Pinescript'), 'pinescript')
            self.content_type_combo.addItem(LANGUAGE_DISPLAY_NAMES.get('auto', 'Auto-Detect'), 'auto')

            # Set default
            default_index = self.content_type_combo.findData(DEFAULT_CONTENT_TYPE)
            if default_index >= 0:
                self.content_type_combo.setCurrentIndex(default_index)

            self.content_type_combo.currentTextChanged.connect(self.on_content_type_change)
            content_type_layout.addWidget(self.content_type_combo)
            content_type_layout.addStretch()
            input_layout.addLayout(content_type_layout)

        # Mode and query row
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Search", "URL"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_change)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter search query or URL")

        self.fetch_button = QPushButton("🔍 Fetch Content")
        self.fetch_button.clicked.connect(self.on_fetch)

        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addWidget(QLabel("Query/URL:"))
        mode_layout.addWidget(self.url_input, 1)
        mode_layout.addWidget(self.fetch_button)
        input_layout.addLayout(mode_layout)

        # Enhanced options (content-type specific)
        self._setup_content_type_options(input_layout)

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

        # Right side: Metadata and insights
        if FREELANCE_MODE or CODE_CATEGORIZATION_ENABLED or PINESCRIPT_ENABLED:
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)

            # Insights panel (content-type aware)
            insights_group = QGroupBox("💡 Insights")
            insights_layout = QVBoxLayout(insights_group)

            self.insights_text = QTextEdit()
            self.insights_text.setReadOnly(True)
            self.insights_text.setMaximumHeight(200)
            self.insights_text.setPlaceholderText("Code insights and recommendations will appear here...")
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
            results_splitter.setSizes([600, 300])

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
            # Content-type specific export formats
            self._update_export_formats()

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
        self.status_label = QLabel("Ready for multi-language RAG content scraping.")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        status_layout.addWidget(self.status_label, 1)
        bottom_layout.addLayout(status_layout)

        main_layout.addWidget(bottom_group)

        # Initialize UI based on current content type
        self.on_content_type_change()

    def _setup_content_type_options(self, parent_layout):
        """Setup content-type specific options."""
        self.options_widget = QWidget()
        self.options_layout = QVBoxLayout(self.options_widget)

        # Python-specific options
        if FREELANCE_MODE:
            self.python_options_widget = QWidget()
            python_options_layout = QHBoxLayout(self.python_options_widget)

            self.freelance_mode_cb = QCheckBox("Freelance Focus")
            self.freelance_mode_cb.setChecked(True)
            self.freelance_mode_cb.setToolTip("Prioritize freelance-relevant code examples")

            self.high_value_only_cb = QCheckBox("High-Value Only")
            self.high_value_only_cb.setToolTip("Show only high client-value code snippets")

            self.include_patterns_cb = QCheckBox("Include Patterns")
            self.include_patterns_cb.setChecked(True)
            self.include_patterns_cb.setToolTip("Include design patterns and best practices")

            python_options_layout.addWidget(self.freelance_mode_cb)
            python_options_layout.addWidget(self.high_value_only_cb)
            python_options_layout.addWidget(self.include_patterns_cb)
            python_options_layout.addStretch()

            self.options_layout.addWidget(self.python_options_widget)

        # NEW: Pinescript-specific options
        if PINESCRIPT_ENABLED:
            self.pinescript_options_widget = QWidget()
            pinescript_options_layout = QHBoxLayout(self.pinescript_options_widget)

            self.trading_focus_cb = QCheckBox("Trading Focus")
            self.trading_focus_cb.setChecked(TRADING_MODE)
            self.trading_focus_cb.setToolTip("Prioritize professional trading strategies and indicators")

            self.include_strategies_cb = QCheckBox("Include Strategies")
            self.include_strategies_cb.setChecked(True)
            self.include_strategies_cb.setToolTip("Include strategy examples and backtesting code")

            self.include_alerts_cb = QCheckBox("Include Alerts")
            self.include_alerts_cb.setChecked(True)
            self.include_alerts_cb.setToolTip("Include alert and notification examples")

            pinescript_options_layout.addWidget(self.trading_focus_cb)
            pinescript_options_layout.addWidget(self.include_strategies_cb)
            pinescript_options_layout.addWidget(self.include_alerts_cb)
            pinescript_options_layout.addStretch()

            self.options_layout.addWidget(self.pinescript_options_widget)
            self.pinescript_options_widget.setVisible(False)  # Hidden by default

        parent_layout.addWidget(self.options_widget)

    def _update_export_formats(self):
        """Update export format combo based on current content type."""
        if not hasattr(self, 'export_format_combo'):
            return

        self.export_format_combo.clear()

        if self.current_content_type == 'pinescript':
            self.export_format_combo.addItems([
                "Standard", "RAG-JSONL", "RAG-Markdown",
                "TradingView-Friendly", "RAG-YAML"
            ])
        else:
            self.export_format_combo.addItems([
                "Standard", "RAG-JSONL", "RAG-Markdown",
                "RAG-XML", "RAG-YAML"
            ])

    def on_content_type_change(self):
        """Handle content type change."""
        if hasattr(self, 'content_type_combo'):
            self.current_content_type = self.content_type_combo.currentData()
        else:
            self.current_content_type = DEFAULT_CONTENT_TYPE

        # Update UI elements based on content type
        self._update_placeholder_text()
        self._update_options_visibility()
        self._update_export_formats()
        self._update_insights_placeholder()

    def _update_placeholder_text(self):
        """Update placeholder text based on content type and mode."""
        mode = self.mode_combo.currentText() if hasattr(self, 'mode_combo') else "Search"

        if mode == "URL":
            self.url_input.setPlaceholderText("Enter URL here (e.g., https://...)")
        else:
            if self.current_content_type == 'pinescript':
                self.url_input.setPlaceholderText(
                    "Enter Pinescript query (e.g., rsi indicator, sma strategy, bollinger bands alert)")
            elif self.current_content_type == 'auto':
                self.url_input.setPlaceholderText(
                    "Enter query (auto-detects Python or Pinescript)")
            else:
                if FREELANCE_MODE:
                    self.url_input.setPlaceholderText(
                        "Enter Python query (e.g., fastapi authentication, stripe integration)")
                else:
                    self.url_input.setPlaceholderText(
                        "Enter module name or search query (e.g., asyncio, pandas http client)")

    def _update_options_visibility(self):
        """Show/hide content-type specific options."""
        if hasattr(self, 'python_options_widget'):
            self.python_options_widget.setVisible(
                self.current_content_type in ['python', 'auto']
            )

        if hasattr(self, 'pinescript_options_widget'):
            self.pinescript_options_widget.setVisible(
                self.current_content_type == 'pinescript'
            )

    def _update_insights_placeholder(self):
        """Update insights placeholder based on content type."""
        if hasattr(self, 'insights_text'):
            if self.current_content_type == 'pinescript':
                self.insights_text.setPlaceholderText(
                    "Trading insights and strategy recommendations will appear here...")
            else:
                self.insights_text.setPlaceholderText(
                    "Freelance insights and recommendations will appear here...")

    def on_mode_change(self, mode):
        self._update_placeholder_text()

    def on_fetch(self):
        query = self.url_input.text().strip()
        mode = self.mode_combo.currentText()

        # Determine content type
        if hasattr(self, 'content_type_combo'):
            selected_content_type = self.content_type_combo.currentData()
        else:
            selected_content_type = DEFAULT_CONTENT_TYPE

        # Auto-detect if needed
        if selected_content_type == 'auto':
            detected_type = detect_content_type(query, self.logger)
            content_type = detected_type
        else:
            content_type = selected_content_type

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
        self.status_label.setText(f"Initializing {content_type} search for: {query}...")
        self.snippets_edit.setPlainText("")

        # Clear analysis panels
        if hasattr(self, 'analysis_edit'):
            self.analysis_edit.setHtml("")
        if hasattr(self, 'insights_text'):
            self.insights_text.setHtml("")
        if hasattr(self, 'categories_text'):
            self.categories_text.setHtml("")
        if hasattr(self, 'suggestions_text'):
            self.suggestions_text.setHtml("")

        self.fetch_worker = FetchWorker(query, mode, content_type, self.logger)
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

        # Display basic snippets with content-type specific formatting
        if self.current_content_type == 'pinescript':
            display_text = "\n\n// -----\n\n".join(self.snippets) if self.snippets else "No Pinescript snippets found."
        else:
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
        """Display enhanced analysis results with content-type awareness."""
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

    def _generate_insights_html(self):
        """Generate HTML for insights with content-type awareness."""
        if not self.enhanced_snippet_data:
            return "<p>No insights available.</p>"

        content_type = self.enhanced_snippet_data[0].get('content_type', 'python')

        if content_type == 'pinescript':
            return self._generate_pinescript_insights_html()
        else:
            return self._generate_python_insights_html()

    def _generate_pinescript_insights_html(self):
        """Generate Pinescript-specific insights."""
        html = "<h4>📈 Trading Value Analysis</h4>"

        # High-value trading snippets
        high_value_snippets = [item for item in self.enhanced_snippet_data
                               if 'high_value' in item['metadata'].get('client_value', '')]

        if high_value_snippets:
            html += f"<p><strong>{len(high_value_snippets)} high-value trading snippets found!</strong></p>"
            html += "<ul>"
            for snippet in high_value_snippets[:3]:
                client_value = snippet['metadata'].get('client_value', '')
                use_cases = snippet['metadata'].get('use_cases', [])
                script_type = snippet['metadata'].get('script_type', 'script')
                html += f"<li>{client_value.replace('_', ' ').title()} ({script_type})"
                if use_cases:
                    html += f" - {', '.join(use_cases[:2])}"
                html += "</li>"
            html += "</ul>"

        # Trading logic analysis
        has_entry_logic = sum(1 for item in self.enhanced_snippet_data
                              if item['metadata'].get('trading_logic', {}).get('has_entry_conditions', False))
        has_risk_mgmt = sum(1 for item in self.enhanced_snippet_data
                            if item['metadata'].get('trading_logic', {}).get('has_risk_management', False))

        if has_entry_logic > 0:
            html += f"<h5>🎯 Trading Logic</h5><ul>"
            html += f"<li>Entry Conditions: {has_entry_logic} snippets</li>"
            html += f"<li>Risk Management: {has_risk_mgmt} snippets</li>"
            html += "</ul>"

        return html

    def _generate_python_insights_html(self):
        """Generate Python-specific insights (existing logic)."""
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

        return html

    def _generate_categories_html(self):
        """Generate categories display with content-type awareness."""
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

        content_type = self.enhanced_snippet_data[0].get('content_type', 'python')

        if content_type == 'pinescript':
            html = "<h4>📂 Trading Categories</h4><ul>"
        else:
            html = "<h4>📂 Category Distribution</h4><ul>"

        for category, count in category_counts.most_common(10):
            html += f"<li><strong>{category.replace('_', ' ').title()}:</strong> {count}</li>"
        html += "</ul>"

        return html

    def _generate_suggestions_html(self):
        """Generate suggestions with content-type awareness."""
        if not self.enhanced_snippet_data:
            return "<p>No suggestions available.</p>"

        content_type = self.enhanced_snippet_data[0].get('content_type', 'python')

        try:
            if content_type == 'pinescript':
                from utils.pinescript_categorizer import PinescriptCategorizer
                categorizer = PinescriptCategorizer()
                first_snippet_meta = self.enhanced_snippet_data[0]['metadata']
                suggestions = categorizer.suggest_related_pinescript_queries(first_snippet_meta)
            else:
                from utils.code_categorizer import CodeCategorizer
                categorizer = CodeCategorizer()
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

    def _generate_analysis_html(self):
        """Generate analysis HTML with content-type awareness."""
        if not self.enhanced_snippet_data:
            return "<p>No analysis data available.</p>"

        content_type = self.enhanced_snippet_data[0].get('content_type', 'python')

        if content_type == 'pinescript':
            html = "<h3>📊 Pinescript Analysis Summary</h3>"
        else:
            html = "<h3>📊 Code Analysis Summary</h3>"

        # Overall statistics
        total_snippets = len(self.enhanced_snippet_data)
        avg_score = sum(item.get('score', 0) for item in self.enhanced_snippet_data) / total_snippets

        if content_type == 'pinescript':
            trading_relevant = sum(1 for item in self.enhanced_snippet_data
                                   if item['metadata'].get('trading_relevant', False))
            html += f"""
            <p><strong>Total Snippets:</strong> {total_snippets}</p>
            <p><strong>Average Trading Score:</strong> {avg_score:.1f}</p>
            <p><strong>Trading Relevant:</strong> {trading_relevant} ({trading_relevant / total_snippets * 100:.1f}%)</p>
            """
        else:
            freelance_relevant = sum(1 for item in self.enhanced_snippet_data
                                     if item['metadata'].get('freelance_relevant', False))
            html += f"""
            <p><strong>Total Snippets:</strong> {total_snippets}</p>
            <p><strong>Average Quality Score:</strong> {avg_score:.1f}</p>
            <p><strong>Freelance Relevant:</strong> {freelance_relevant} ({freelance_relevant / total_snippets * 100:.1f}%)</p>
            """

        return html

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
        self.status_label.setText(f"Saving {len(self.snippets)} {self.current_content_type} snippets...")

        self.save_worker = SaveWorker(self.snippets, directory, self.current_content_type, self.logger)
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
        self.progress_bar.setRange(0, 100)

    def handle_save_error(self, error_message):
        self.logger.error(f"GUI received save error summary: {error_message}")
        self.status_label.setText(f"Save Error. See log for details.")
        self.save_button.setEnabled(True)
        self.fetch_button.setEnabled(True)
        if self.snippets:
            if hasattr(self, 'rag_export_button'):
                self.rag_export_button.setEnabled(True)
            if hasattr(self, 'dual_llm_export_button'):
                self.dual_llm_export_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_rag_export(self):
        """Handle RAG export with content-type awareness."""
        if not self.snippets:
            self.status_label.setText("No snippets to export.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory for RAG Export")
        if not directory:
            self.status_label.setText("Export cancelled.")
            return

        try:
            selected_format = self.export_format_combo.currentText()

            if selected_format == "Standard":
                self.on_save()
                return

            # Determine appropriate exporter based on content type
            if self.current_content_type == 'pinescript':
                # Use Pinescript-specific export logic
                from storage.rag_exporter import RAGExporter
                exporter = RAGExporter()

                format_map = {
                    "RAG-JSONL": "jsonl",
                    "RAG-Markdown": "markdown",
                    "TradingView-Friendly": "tradingview",
                    "RAG-YAML": "yaml"
                }
            else:
                # Use existing Python export logic
                from storage.rag_exporter import RAGExporter
                exporter = RAGExporter()

                format_map = {
                    "RAG-JSONL": "jsonl",
                    "RAG-Markdown": "markdown",
                    "RAG-XML": "xml",
                    "RAG-YAML": "yaml"
                }

            # Use enhanced data if available
            if hasattr(self, 'enhanced_snippet_data') and self.enhanced_snippet_data:
                snippets_data = self.enhanced_snippet_data
            else:
                # Create basic data structure
                snippets_data = []
                for snippet in self.snippets:
                    snippets_data.append({
                        'code': snippet,
                        'score': 5,
                        'metadata': {'content_type': self.current_content_type}
                    })

            query = self.url_input.text().strip() or "search_query"
            export_format = format_map.get(selected_format, "jsonl")

            output_file = exporter.export_for_rag(snippets_data, directory, query, export_format)
            self.status_label.setText(f"RAG export complete: {output_file}")

        except ImportError:
            self.status_label.setText("Enhanced export features not available.")
        except Exception as e:
            self.logger.error(f"RAG export error: {e}")
            self.status_label.setText(f"RAG export failed: {str(e)}")

    def on_dual_llm_export(self):
        """Handle dual LLM export with content-type awareness."""
        if not self.snippets:
            self.status_label.setText("No snippets to export.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Dual LLM Export")
        if not directory:
            self.status_label.setText("Export cancelled.")
            return

        try:
            from storage.embedding_rag_exporter import EmbeddingRAGExporter

            # Enhanced categorization based on content type
            if self.current_content_type == 'pinescript':
                from utils.pinescript_categorizer import PinescriptCategorizer
                categorizer = PinescriptCategorizer()
            else:
                from utils.code_categorizer import CodeCategorizer
                categorizer = CodeCategorizer()

            enhanced_data = []

            if hasattr(self, 'enhanced_snippet_data') and self.enhanced_snippet_data:
                enhanced_data = self.enhanced_snippet_data
            else:
                # Create enhanced categorization
                for snippet in self.snippets:
                    if self.current_content_type == 'pinescript':
                        categorization = categorizer.categorize_pinescript(snippet)
                        score = categorization.get('trading_value_score', 0) + 5
                    else:
                        categorization = categorizer.categorize_snippet(snippet)
                        score = categorization.get('freelance_score', 0) + 5

                    enhanced_data.append({
                        'code': snippet,
                        'score': score,
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
        # Clean up workers
        if hasattr(self, 'fetch_worker') and self.fetch_worker.isRunning():
            self.logger.info("Attempting to quit fetch_worker...")
            self.fetch_worker.quit()
            if not self.fetch_worker.wait(3000):
                self.logger.warning("Fetch worker did not terminate gracefully, forcing termination.")
                self.fetch_worker.terminate()
                self.fetch_worker.wait()
        if hasattr(self, 'save_worker') and self.save_worker.isRunning():
            self.logger.info("Attempting to quit save_worker...")
            self.save_worker.quit()
            if not self.save_worker.wait(3000):
                self.logger.warning("Save worker did not terminate gracefully, forcing termination.")
                self.save_worker.terminate()
                self.save_worker.wait()
        event.accept()


# Create alias for backward compatibility
MainWindow = EnhancedMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Load stylesheet
    if STYLESHEET_PATH:
        try:
            with open(STYLESHEET_PATH, encoding="utf-8") as f:
                app.setStyleSheet(f.read())
            print(f"Stylesheet '{STYLESHEET_PATH}' loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: Stylesheet not found at '{STYLESHEET_PATH}'. Using default styles.")
        except Exception as e:
            print(f"Could not load stylesheet from '{STYLESHEET_PATH}': {e}")
    else:
        print("No stylesheet path configured. Using default styles.")

    window = EnhancedMainWindow()
    window.show()
    sys.exit(app.exec())