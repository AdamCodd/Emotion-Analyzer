# Emotion Analyzer

## Description

Emotion Analyzer is a Python-based text analysis tool designed to identify emotional tones in given text. It uses a lexicon-based approach and allows for the incorporation of booster words and negations to provide a more nuanced understanding of the text's emotional content. The application can read from various file formats like CSV, JSON, JSONL, and plain text. It also supports multi-threading for optimized performance.

## Features

- Lexicon-based emotion analysis
- Customizable lexicon and encoding
- Booster and negation word handling
- Ambiguity threshold for detecting mixed emotions
- Supports multiple input formats: CSV, JSON, JSONL, TXT
- Multi-threading support
- Export results to CSV, JSON, JSONL, TXT

### Lexicon

The lexicon used in this project is derived from the "NRC-Emotion-Intensity-Lexicon-v1". It has been heavily curated, deduplicated, and augmented with the "lemminflect" library to match six emotions: "sadness", "love", "joy", "anger", "fear", and "surprise". The lexicon is still a work in progress.

## Requirements

- Python 3.x

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/emotion-analyzer.git
```

Navigate into the project directory:

```bash
cd emotion-analyzer
```

## Usage

### Command Line Interface

The script can be run from the command line using `argparse`. 

```bash
python main.py --lexicon_path=path/to/lexicon.csv --input_file=path/to/input.txt --output=output_file_name --format=csv
```

#### Arguments

- `--lexicon_path`: Path to the lexicon file. **(Required)**
- `--max_threads`: Maximum number of threads for parallel processing. (Default: 4)
- `--encoding`: File encoding for the lexicon (default is utf-8).
- `--ambiguity_threshold`: Threshold for ambiguous emotions. (Default: 0.1)
- `--input_file`: Input file to read texts from (CSV, TXT, JSON, JSONL supported).
- `--json_key`: Key to read texts from when the input file is JSON or JSONL. (Default: 'text')
- `--output`: Base name of the output file to save the results. (Default: 'output')
- `--format`: Output format: csv, txt, json, jsonl. (Default: csv)
- `--text`: Text to analyze if not reading from a file.

### Code API

You can also use Emotion Analyzer in your Python project. Import the `EmotionAnalyzer` and `ResultsSaver` classes from the code and use as shown in the examples in the source code.

## License

MIT License. See [LICENSE](LICENSE) for more details.

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.