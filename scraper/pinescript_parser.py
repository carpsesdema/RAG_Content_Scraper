# scraper/pinescript_parser.py
import code
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PinescriptFunction:
    """Represents a function definition in Pinescript."""
    name: str
    parameters: List[str]
    return_type: Optional[str]
    body: str
    line_start: int
    line_end: int
    is_exported: bool = False


@dataclass
class PinescriptInput:
    """Represents an input parameter in Pinescript."""
    name: str
    input_type: str
    default_value: str
    title: str
    constraints: Dict[str, str]


@dataclass
class PinescriptPlot:
    """Represents a plot statement in Pinescript."""
    variable: str
    title: str
    color: Optional[str]
    style: Optional[str]
    linewidth: Optional[str]


@dataclass
class PinescriptAlert:
    """Represents an alert condition in Pinescript."""
    condition: str
    title: str
    message: Optional[str]


class PinescriptParser:
    """Parser for Pinescript code to extract structural information."""

    def __init__(self):
        # Pinescript version pattern
        self.version_pattern = re.compile(r'//@version\s*=\s*([0-9]+)')

        # Script declaration patterns
        self.indicator_pattern = re.compile(r'indicator\s*\(\s*["\']([^"\']*)["\']')
        self.strategy_pattern = re.compile(r'strategy\s*\(\s*["\']([^"\']*)["\']')
        self.library_pattern = re.compile(r'library\s*\(\s*["\']([^"\']*)["\']')

        # Function definition patterns
        self.function_pattern = re.compile(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*=>?\s*$', re.MULTILINE)
        self.export_function_pattern = re.compile(r'^(\s*)export\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*=>?\s*$',
                                                  re.MULTILINE)

        # Input patterns
        self.input_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*input(?:\.([a-zA-Z]+))?\s*\((.*?)\)')

        # Plot patterns
        self.plot_pattern = re.compile(r'plot\s*\(\s*([^,]+)(?:,\s*(.+?))?\s*\)')

        # Alert patterns
        self.alert_pattern = re.compile(r'alertcondition\s*\(\s*([^,]+)(?:,\s*title\s*=\s*["\']([^"\']*)["\'])?')
        self.alert_simple_pattern = re.compile(r'alert\s*\(\s*["\']([^"\']*)["\']')

        # Strategy patterns
        self.strategy_entry_pattern = re.compile(r'strategy\.entry\s*\(\s*["\']([^"\']*)["\']')
        self.strategy_exit_pattern = re.compile(r'strategy\.exit\s*\(\s*["\']([^"\']*)["\']')
        self.strategy_close_pattern = re.compile(r'strategy\.close\s*\(\s*["\']([^"\']*)["\']')

        # Technical analysis patterns
        self.ta_pattern = re.compile(r'ta\.([a-zA-Z_][a-zA-Z0-9_]*)')
        self.math_pattern = re.compile(r'math\.([a-zA-Z_][a-zA-Z0-9_]*)')

        # Variable declaration patterns
        self.var_pattern = re.compile(r'(var|varip)\s+([a-zA-Z_][a-zA-Z0-9_]*)')

    def parse_pinescript(self, code: str) -> Dict:
        """Parse Pinescript code and extract structural information."""
        lines = code.split('\n')

        result = {
            'version': self._extract_version(code),
            'script_type': self._extract_script_type(code),
            'script_title': self._extract_script_title(code),
            'functions': self._extract_functions(code, lines),
            'inputs': self._extract_inputs(code),
            'plots': self._extract_plots(code),
            'alerts': self._extract_alerts(code),
            'strategy_calls': self._extract_strategy_calls(code),
            'ta_functions': self._extract_ta_functions(code),
            'variables': self._extract_variables(code),
            'imports': self._extract_imports(code),
            'annotations': self._extract_annotations(code),
            'complexity_metrics': self._calculate_complexity(code, lines)
        }

        return result

    def _extract_version(self, code: str) -> Optional[str]:
        """Extract Pinescript version."""
        match = self.version_pattern.search(code)
        return f"v{match.group(1)}" if match else None

    def _extract_script_type(self, code: str) -> str:
        """Extract script type (indicator, strategy, library)."""
        if self.library_pattern.search(code):
            return 'library'
        elif self.strategy_pattern.search(code):
            return 'strategy'
        elif self.indicator_pattern.search(code):
            return 'indicator'
        else:
            return 'script'

    def _extract_script_title(self, code: str) -> Optional[str]:
        """Extract script title from declaration."""
        for pattern in [self.indicator_pattern, self.strategy_pattern, self.library_pattern]:
            match = pattern.search(code)
            if match:
                return match.group(1)
        return None

    def _extract_functions(self, code: str, lines: List[str]) -> List[PinescriptFunction]:
        """Extract user-defined functions."""
        functions = []

        # Regular functions
        for match in self.function_pattern.finditer(code):
            function = self._parse_function_match(match, lines, False)
            if function:
                functions.append(function)

        # Exported functions
        for match in self.export_function_pattern.finditer(code):
            function = self._parse_function_match(match, lines, True)
            if function:
                functions.append(function)

        return functions

    def _parse_function_match(self, match, lines: List[str], is_exported: bool) -> Optional[PinescriptFunction]:
        """Parse a function match into a PinescriptFunction object."""
        try:
            indent = match.group(1)
            name = match.group(2)
            params_str = match.group(3)

            # Parse parameters
            parameters = []
            if params_str.strip():
                # Simple parameter parsing - could be enhanced
                params = [p.strip() for p in params_str.split(',')]
                parameters = [p for p in params if p]

            # Find function body (this is simplified)
            start_line = code[:match.start()].count('\n')
            end_line = start_line + 1

            # Look for the function body (simplified heuristic)
            body_lines = []
            current_line = start_line + 1
            base_indent_len = len(indent)

            while current_line < len(lines):
                line = lines[current_line]
                if line.strip() == '':
                    current_line += 1
                    continue

                line_indent = len(line) - len(line.lstrip())
                if line_indent <= base_indent_len and line.strip():
                    break

                body_lines.append(line)
                current_line += 1
                end_line = current_line

            body = '\n'.join(body_lines)

            return PinescriptFunction(
                name=name,
                parameters=parameters,
                return_type=None,  # Pinescript doesn't have explicit return types
                body=body,
                line_start=start_line,
                line_end=end_line,
                is_exported=is_exported
            )
        except Exception:
            return None

    def _extract_inputs(self, code: str) -> List[PinescriptInput]:
        """Extract input parameters."""
        inputs = []

        for match in self.input_pattern.finditer(code):
            try:
                name = match.group(1)
                input_type = match.group(2) or 'basic'
                params_str = match.group(3)

                # Parse input parameters
                default_value = ""
                title = ""
                constraints = {}

                # Simple parameter parsing
                if params_str:
                    # Extract default value (first parameter)
                    params = [p.strip() for p in params_str.split(',')]
                    if params:
                        default_value = params[0]

                    # Look for title
                    title_match = re.search(r'title\s*=\s*["\']([^"\']*)["\']', params_str)
                    if title_match:
                        title = title_match.group(1)

                    # Look for constraints
                    for constraint in ['minval', 'maxval', 'step']:
                        constraint_match = re.search(f'{constraint}\\s*=\\s*([^,)]+)', params_str)
                        if constraint_match:
                            constraints[constraint] = constraint_match.group(1).strip()

                inputs.append(PinescriptInput(
                    name=name,
                    input_type=input_type,
                    default_value=default_value,
                    title=title,
                    constraints=constraints
                ))
            except Exception:
                continue

        return inputs

    def _extract_plots(self, code: str) -> List[PinescriptPlot]:
        """Extract plot statements."""
        plots = []

        for match in self.plot_pattern.finditer(code):
            try:
                variable = match.group(1).strip()
                params_str = match.group(2) or ""

                # Parse plot parameters
                title = ""
                color = None
                style = None
                linewidth = None

                if params_str:
                    # Extract title
                    title_match = re.search(r'title\s*=\s*["\']([^"\']*)["\']', params_str)
                    if title_match:
                        title = title_match.group(1)

                    # Extract color
                    color_match = re.search(r'color\s*=\s*([^,)]+)', params_str)
                    if color_match:
                        color = color_match.group(1).strip()

                    # Extract style
                    style_match = re.search(r'style\s*=\s*([^,)]+)', params_str)
                    if style_match:
                        style = style_match.group(1).strip()

                    # Extract linewidth
                    linewidth_match = re.search(r'linewidth\s*=\s*([^,)]+)', params_str)
                    if linewidth_match:
                        linewidth = linewidth_match.group(1).strip()

                plots.append(PinescriptPlot(
                    variable=variable,
                    title=title,
                    color=color,
                    style=style,
                    linewidth=linewidth
                ))
            except Exception:
                continue

        return plots

    def _extract_alerts(self, code: str) -> List[PinescriptAlert]:
        """Extract alert conditions."""
        alerts = []

        # alertcondition() alerts
        for match in self.alert_pattern.finditer(code):
            try:
                condition = match.group(1).strip()
                title = match.group(2) or ""

                alerts.append(PinescriptAlert(
                    condition=condition,
                    title=title,
                    message=None
                ))
            except Exception:
                continue

        # Simple alert() calls
        for match in self.alert_simple_pattern.finditer(code):
            try:
                message = match.group(1)

                alerts.append(PinescriptAlert(
                    condition="",
                    title="",
                    message=message
                ))
            except Exception:
                continue

        return alerts

    def _extract_strategy_calls(self, code: str) -> Dict[str, List[str]]:
        """Extract strategy function calls."""
        strategy_calls = {
            'entries': [],
            'exits': [],
            'closes': []
        }

        # Entry calls
        for match in self.strategy_entry_pattern.finditer(code):
            strategy_calls['entries'].append(match.group(1))

        # Exit calls
        for match in self.strategy_exit_pattern.finditer(code):
            strategy_calls['exits'].append(match.group(1))

        # Close calls
        for match in self.strategy_close_pattern.finditer(code):
            strategy_calls['closes'].append(match.group(1))

        return strategy_calls

    def _extract_ta_functions(self, code: str) -> List[str]:
        """Extract technical analysis function calls."""
        ta_functions = []

        for match in self.ta_pattern.finditer(code):
            function_name = match.group(1)
            if function_name not in ta_functions:
                ta_functions.append(function_name)

        return ta_functions

    def _extract_variables(self, code: str) -> List[Tuple[str, str]]:
        """Extract variable declarations."""
        variables = []

        for match in self.var_pattern.finditer(code):
            var_type = match.group(1)  # 'var' or 'varip'
            var_name = match.group(2)
            variables.append((var_type, var_name))

        return variables

    def _extract_imports(self, code: str) -> List[str]:
        """Extract library imports."""
        imports = []

        # Look for library import statements
        import_pattern = re.compile(r'import\s+([a-zA-Z_][a-zA-Z0-9_./]*)')
        for match in import_pattern.finditer(code):
            imports.append(match.group(1))

        return imports

    def _extract_annotations(self, code: str) -> Dict[str, List[str]]:
        """Extract annotations and comments."""
        annotations = {
            'comments': [],
            'descriptions': [],
            'functions': []
        }

        lines = code.split('\n')
        for line in lines:
            stripped = line.strip()

            # Single line comments
            if stripped.startswith('//'):
                comment = stripped[2:].strip()

                # Check for special annotations
                if comment.startswith('@description'):
                    annotations['descriptions'].append(comment[12:].strip())
                elif comment.startswith('@function'):
                    annotations['functions'].append(comment[9:].strip())
                else:
                    annotations['comments'].append(comment)

        return annotations

    def _calculate_complexity(self, code: str, lines: List[str]) -> Dict[str, int]:
        """Calculate complexity metrics."""
        metrics = {
            'total_lines': len(lines),
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'if_statements': 0,
            'for_loops': 0,
            'function_calls': 0,
            'nested_blocks': 0
        }

        # Line counting
        for line in lines:
            stripped = line.strip()
            if not stripped:
                metrics['blank_lines'] += 1
            elif stripped.startswith('//'):
                metrics['comment_lines'] += 1
            else:
                metrics['code_lines'] += 1

        # Control structure counting
        metrics['if_statements'] = len(re.findall(r'\bif\b', code))
        metrics['for_loops'] = len(re.findall(r'\bfor\b', code))

        # Function call counting (simplified)
        metrics['function_calls'] = len(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(', code))

        # Nested block estimation (simplified)
        max_indent = 0
        current_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > current_indent:
                    current_indent = indent
                    max_indent = max(max_indent, indent // 4)  # Assuming 4-space indents

        metrics['nested_blocks'] = max_indent

        return metrics

    def extract_semantic_blocks(self, code: str) -> List[Dict]:
        """Extract semantic blocks of code (functions, indicators, strategies)."""
        parsed = self.parse_pinescript(code)
        blocks = []

        # Add script header as a block
        script_type = parsed['script_type']
        script_title = parsed['script_title']

        if script_title:
            blocks.append({
                'type': 'script_declaration',
                'subtype': script_type,
                'title': script_title,
                'content': self._extract_script_declaration(code),
                'metadata': {
                    'version': parsed['version'],
                    'script_type': script_type
                }
            })

        # Add input section
        if parsed['inputs']:
            input_content = self._extract_input_section(code)
            if input_content:
                blocks.append({
                    'type': 'inputs',
                    'title': 'Input Parameters',
                    'content': input_content,
                    'metadata': {
                        'input_count': len(parsed['inputs']),
                        'inputs': parsed['inputs']
                    }
                })

        # Add functions
        for func in parsed['functions']:
            blocks.append({
                'type': 'function',
                'title': func.name,
                'content': self._reconstruct_function(func),
                'metadata': {
                    'parameters': func.parameters,
                    'is_exported': func.is_exported,
                    'line_range': (func.line_start, func.line_end)
                }
            })

        # Add plotting section
        if parsed['plots']:
            plot_content = self._extract_plot_section(code)
            if plot_content:
                blocks.append({
                    'type': 'plotting',
                    'title': 'Plotting',
                    'content': plot_content,
                    'metadata': {
                        'plot_count': len(parsed['plots']),
                        'plots': parsed['plots']
                    }
                })

        # Add alert section
        if parsed['alerts']:
            alert_content = self._extract_alert_section(code)
            if alert_content:
                blocks.append({
                    'type': 'alerts',
                    'title': 'Alerts',
                    'content': alert_content,
                    'metadata': {
                        'alert_count': len(parsed['alerts']),
                        'alerts': parsed['alerts']
                    }
                })

        return blocks

    def _extract_script_declaration(self, code: str) -> str:
        """Extract the script declaration line."""
        lines = code.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['indicator(', 'strategy(', 'library(']):
                return line.strip()
        return ""

    def _extract_input_section(self, code: str) -> str:
        """Extract all input-related lines."""
        lines = code.split('\n')
        input_lines = []

        for line in lines:
            if 'input' in line and '=' in line:
                input_lines.append(line.strip())

        return '\n'.join(input_lines)

    def _extract_plot_section(self, code: str) -> str:
        """Extract all plot-related lines."""
        lines = code.split('\n')
        plot_lines = []

        for line in lines:
            if line.strip().startswith('plot(') or 'plot(' in line:
                plot_lines.append(line.strip())

        return '\n'.join(plot_lines)

    def _extract_alert_section(self, code: str) -> str:
        """Extract all alert-related lines."""
        lines = code.split('\n')
        alert_lines = []

        for line in lines:
            if 'alertcondition(' in line or 'alert(' in line:
                alert_lines.append(line.strip())

        return '\n'.join(alert_lines)

    def _reconstruct_function(self, func: PinescriptFunction) -> str:
        """Reconstruct function code from PinescriptFunction object."""
        params_str = ', '.join(func.parameters) if func.parameters else ''
        export_prefix = 'export ' if func.is_exported else ''

        return f"{export_prefix}{func.name}({params_str}) =>\n{func.body}"


def extract_pinescript_code(content: str) -> List[str]:
    """
    Extract Pinescript code blocks from content.
    Enhanced version of the general extract_code function.
    """
    snippets = []

    # First, try to parse as a complete Pinescript
    if is_pinescript_code(content):
        # Parse into semantic blocks for better organization
        parser = PinescriptParser()
        try:
            blocks = parser.extract_semantic_blocks(content)

            # Add full script as main snippet
            snippets.append(content.strip())

            # Add individual blocks as separate snippets
            for block in blocks:
                if block['type'] in ['function', 'inputs', 'plotting', 'alerts']:
                    block_content = block['content']
                    if block_content and len(block_content.strip()) > 20:
                        snippets.append(block_content)

        except Exception:
            # Fallback to basic extraction
            snippets.append(content.strip())

    else:
        # Look for embedded Pinescript in markdown or HTML
        # Fenced code blocks with pinescript language
        import re

        fence_patterns = [
            re.compile(r'```(?:pinescript|pine)\n(.*?)```', re.DOTALL | re.IGNORECASE),
            re.compile(r'```\n(.*?)```', re.DOTALL),  # Generic code blocks
        ]

        for pattern in fence_patterns:
            for match in pattern.finditer(content):
                code_block = match.group(1).strip()
                if is_pinescript_code(code_block):
                    snippets.append(code_block)

        # Look for HTML code blocks
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(content, 'html.parser')
            for code_tag in soup.find_all(['code', 'pre']):
                code_text = code_tag.get_text().strip()
                if is_pinescript_code(code_text):
                    snippets.append(code_text)
        except Exception:
            pass

    # Deduplicate while preserving order
    seen = set()
    unique_snippets = []
    for snippet in snippets:
        if snippet not in seen and len(snippet.strip()) > 10:
            seen.add(snippet)
            unique_snippets.append(snippet)

    return unique_snippets


def is_pinescript_code(code: str) -> bool:
    """Check if code appears to be Pinescript."""
    if not code or len(code.strip()) < 10:
        return False

    code_lower = code.lower()

    # Strong Pinescript indicators
    strong_indicators = [
        '//@version',
        'indicator(',
        'strategy(',
        'library(',
        'ta.',
        'strategy.entry',
        'strategy.exit',
        'alertcondition(',
        'plot(',
        'hline(',
        'bgcolor(',
        'plotshape(',
        'plotcandle('
    ]

    # Weak indicators (need multiple)
    weak_indicators = [
        'close', 'high', 'low', 'open', 'volume',
        'input.', 'math.', 'request.',
        'color.', 'line.', 'label.',
        'var ', 'varip ', '=>'
    ]

    strong_count = sum(1 for indicator in strong_indicators if indicator in code_lower)
    weak_count = sum(1 for indicator in weak_indicators if indicator in code_lower)

    # Must have at least one strong indicator OR multiple weak indicators
    return strong_count > 0 or weak_count >= 3


def clean_pinescript_snippet(snippet: str) -> str:
    """Clean Pinescript snippet by removing common artifacts."""
    if not snippet:
        return ""

    lines = snippet.splitlines()
    cleaned_lines = []

    for line in lines:
        # Remove TradingView UI artifacts
        if line.strip().startswith('Copy code'):
            continue
        if 'Open in Pine Editor' in line:
            continue
        if 'Add to Chart' in line:
            continue

        # Clean line prefixes from forums/docs
        if line.startswith(">>> "):
            cleaned_lines.append(line[4:])
        elif line.startswith("... "):
            cleaned_lines.append(line[4:])
        else:
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Remove common copy artifacts
    cleaned_text = re.sub(r'\bCopy code\b', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\bView on TradingView\b', '', cleaned_text, flags=re.IGNORECASE)

    return cleaned_text.strip()