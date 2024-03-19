# Emotion Analyzer

## Description

Emotion Analyzer is a Python-based tool designed to identify emotional tones in text. Utilizing a lexicon-based approach, it incorporates booster words and negations to offer a nuanced analysis of emotional content. The tool supports various file formats such as CSV, JSON, JSONL, and plain text, and is optimized for performance through multi-threading. Its capabilities can be further enhanced when used in conjunction with an [emotion classifier](https://huggingface.co/AdamCodd/distilbert-base-uncased-finetuned-emotion-balanced).

## Features

- Lexicon-based emotion analysis
- Customizable lexicon and encoding
- Booster and negation word handling
- Ambiguity threshold for detecting mixed emotions
- Supports multiple input formats: CSV, JSON, JSONL, TXT
- Multi-threading support
- Export results to CSV, JSON, JSONL, TXT

### Lexicon

The lexicon used in this project originates from the [NRC-Emotion-Intensity-Lexicon-v1](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm). It has undergone extensive curation and deduplication, and has been augmented with the [lemminflect](https://github.com/bjascob/LemmInflect) library to align with six fundamental emotions: "sadness," "love," "joy," "anger," "fear," and "surprise." The lexicon, which contains 9,762 entries, is functional but remains a work in progress.

## Default Lexicon Path

By default, the program looks for a `lexicon.csv` file in the same directory as the script. You can override this by specifying a different path using the `--lexicon_path` command-line argument.

## Requirements

- Python 3.6+

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
