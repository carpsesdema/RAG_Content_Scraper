# RAG Content Scraper - Enhanced for Freelance Python Development

A powerful tool for collecting high-quality Python code snippets optimized for RAG (Retrieval-Augmented Generation) systems and dual LLM workflows. Specifically enhanced for freelance Python developers working with modern tech stacks.

## 🚀 New Enhanced Features

### 🎯 Freelance-Focused Content
- **High-Value Code Detection**: Automatically identifies code with high client value (payments, APIs, integrations)
- **Modern Tech Stack**: FastAPI, Stripe, Twilio, AWS, Docker examples prioritized
- **Client Integration Patterns**: Payment processing, communication services, cloud deployments
- **Practical Examples**: Ready-to-use code for common freelance projects

### 🧠 Dual LLM Optimization
- **Chat LLM Format**: Natural language descriptions for instruction-giving
- **Code LLM Format**: Pure code with minimal context for implementation
- **Cross-Reference Index**: Links between chat and code formats
- **Embedding-Ready Chunks**: Pre-optimized for vector databases

### 📊 Smart Content Processing
- **Quality Filtering**: Scores code based on documentation, complexity, and usefulness
- **Smart Deduplication**: AST-based similarity detection beyond simple text matching
- **Code Categorization**: Automatic tagging by patterns, complexity, and use cases
- **Freelance Value Scoring**: Ranks code by potential client value

### 📁 Multiple Export Formats
- **Standard**: Traditional text files
- **RAG-JSONL**: Line-delimited JSON for embedding systems
- **RAG-Markdown**: Structured documentation format
- **RAG-XML/YAML**: Structured data formats
- **Dual LLM**: Separate optimized files for chat and code LLMs

## 🛠 Installation

### Prerequisites
- Python 3.11+
- PySide6 for GUI
- GitHub token (recommended for better API limits)

### Quick Setup
```bash
# Clone and install
git clone <your-repo>
cd RAG_Content_Scraper
pip install -r requirements.txt

# Set GitHub token (optional but recommended)
export GITHUB_TOKEN="your_github_token_here"

# Test installation
python test_enhanced_features.py

# Run the application
python main.py
```

### Dependencies
```txt
# Core requirements
PySide6          # GUI framework
requests         # HTTP requests
beautifulsoup4   # HTML parsing
lxml            # XML parsing
PyGithub        # GitHub API
pyyaml          # YAML export

# Optional enhancements
pandas          # Data processing
numpy           # Numerical operations
```

## 🎮 Usage

### Basic Usage
1. **Launch**: Run `python main.py`
2. **Search**: Enter queries like "fastapi authentication" or "stripe payment"
3. **Export**: Choose from multiple RAG-optimized formats

### Advanced Features

#### Freelance Mode
- Enable "Freelance Focus" for client-relevant code
- Use "High-Value Only" to filter premium content
- Search with freelance-specific queries:
  - `fastapi stripe integration`
  - `twilio sms automation`
  - `docker deployment fastapi`
  - `pytest testing patterns`

#### Dual LLM Export
Perfect for hybrid chat + code LLM systems:
1. Perform search with enhanced categorization
2. Click "Dual LLM Export"
3. Get optimized files:
   - `chat_llm_*.jsonl` - Natural language for chat LLM
   - `code_llm_*.jsonl` - Pure code for code LLM
   - `unified_*.json` - Cross-reference index
   - `vector_db_*.jsonl` - Embedding-ready format

#### Query Optimization
The system automatically:
- Expands queries based on freelance relevance
- Prioritizes high-value sources (GitHub files > README > docs)
- Applies smart deduplication
- Categorizes by complexity and use case

## 📚 Enhanced Sources

### Standard Sources
- **Python stdlib docs**: Official documentation
- **Stack Overflow**: Community Q&A with answers
- **GitHub READMEs**: Project documentation examples
- **GitHub Files**: Real implementation code

### Freelance-Specific Sources  
- **FastAPI Examples**: Modern API development patterns
- **Integration Examples**: Stripe, Twilio, SendGrid, AWS
- **Automation Scripts**: File processing, web scraping, scheduling
- **Testing Patterns**: pytest, fixtures, mocking
- **Deployment Examples**: Docker, docker-compose, CI/CD

### Quality Enhancements
- **AST Analysis**: Extracts functions, classes, patterns
- **Dependency Discovery**: Finds related packages from requirements.txt
- **Pattern Recognition**: Identifies design patterns and best practices
- **Complexity Scoring**: Rates code difficulty and usefulness

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Freelance mode
FREELANCE_MODE = True
PRIORITIZE_PRACTICAL_EXAMPLES = True
BOOST_FASTAPI_CONTENT = 2.0
BOOST_CLIENT_INTEGRATIONS = 2.5

# Quality filtering
QUALITY_FILTER_ENABLED = True
MIN_SNIPPET_QUALITY_SCORE = 3
SMART_DEDUPLICATION_ENABLED = True

# Enhanced exports
EMBEDDING_RAG_EXPORT_ENABLED = True
DUAL_LLM_EXPORT = True
CODE_CATEGORIZATION_ENABLED = True
```

## 🔧 Technical Architecture

### Core Components
- **Searcher**: Multi-source content fetching with enhanced crawling
- **Quality Filter**: AST-based code analysis and scoring
- **Categorizer**: Pattern recognition and freelance value assessment
- **Deduplicator**: Smart similarity detection
- **Exporters**: Multiple RAG-optimized output formats

### Data Flow
```
Query → Multi-Source Fetch → Quality Filter → Categorization → 
Deduplication → Export (Standard/RAG/Dual LLM)
```

### Enhanced Processing Pipeline
1. **Source Prioritization**: Weight sources by freelance relevance
2. **Content Analysis**: AST parsing, pattern detection, complexity scoring
3. **Quality Filtering**: Remove low-value snippets
4. **Smart Deduplication**: Structural similarity detection
5. **Categorization**: Tag by use case, complexity, client value
6. **Export Optimization**: Format for specific LLM consumption

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_enhanced_features.py
```

Tests cover:
- Configuration loading
- Code categorization accuracy
- Deduplication effectiveness
- Export format validation
- Freelancer source functionality
- Component integration

## 📊 Output Formats

### Standard Exports
- **Text Files**: Individual `.txt` files per snippet
- **RAG-JSONL**: Line-delimited JSON for vector databases
- **RAG-Markdown**: Documentation-style with metadata
- **RAG-XML/YAML**: Structured data formats

### Dual LLM Exports
- **Chat LLM File**: Natural language explanations, use cases, tips
- **Code LLM File**: Pure code with minimal context, chunked optimally
- **Unified Index**: Cross-references between formats
- **Vector DB File**: Embedding-ready text combining code + context
- **Embedding Chunks**: Pre-chunked for optimal embedding generation

### Metadata Included
- Source information and URLs
- Quality scores and complexity ratings
- Freelance value assessments
- Code patterns and categories
- Implementation tips and pitfalls
- Related query suggestions

## 🎯 Freelance Use Cases

### High-Value Searches
- `stripe subscription management`
- `twilio sms verification`
- `fastapi jwt authentication`
- `aws s3 file upload`
- `docker production deployment`
- `pytest api testing`

### Project Types
- **API Development**: FastAPI, authentication, validation
- **Payment Processing**: Stripe integration, webhooks
- **Communication**: Email, SMS, Slack integrations
- **Data Processing**: ETL pipelines, CSV/Excel automation
- **DevOps**: Docker, CI/CD, monitoring
- **Testing**: Automated testing, mocking, fixtures

## 🚨 Troubleshooting

### Common Issues
1. **Rate Limiting**: Set `GITHUB_TOKEN` environment variable
2. **No Results**: Try broader queries like "fastapi" instead of specific functions
3. **Import Errors**: Install missing dependencies with `pip install -r requirements.txt`
4. **Quality Filter Too Strict**: Lower `MIN_SNIPPET_QUALITY_SCORE` in config

### Debug Mode
Enable verbose logging:
```python
# In config.py
DEBUG_MODE = True
VERBOSE_LOGGING = True
LOG_LEVEL_CONSOLE = "DEBUG"
```

## 🤝 Contributing

### Adding New Sources
1. Create new source class in `scraper/`
2. Implement required methods with error handling
3. Add to `search_additional_sources()` or `search_freelancer_sources()`
4. Update configuration options

### Enhancing Categorization
1. Modify `utils/code_categorizer.py`
2. Add new categories to `self.categories`
3. Update freelance indicators
4. Test with `test_enhanced_features.py`

## 📄 License

[Your license here]

## 🙏 Acknowledgments

- Built for freelance Python developers
- Optimized for modern tech stacks
- Designed for dual LLM workflows
- Enhanced for RAG systems

---

**Perfect for freelancers working with**: FastAPI • Stripe • Twilio • AWS • Docker • pytest • Modern Python stacks

**Optimized for**: RAG systems • Vector databases • Dual LLM workflows • Code embeddings • Freelance development