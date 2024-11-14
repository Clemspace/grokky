# Grokky

A comprehensive study of the grokking phenomenon in transformer architectures. This project implements different transformer architectures and analyzes their learning dynamics, with a particular focus on understanding when and how grokking occurs.

## Project Structure
\`\`\`
src/
├── data/            # Data handling and dataset implementations
├── models/          # Model architectures and configurations
├── training/        # Training infrastructure
├── analysis/        # Analysis and visualization tools
└── utils/          # Utility functions and logging
\`\`\`

## Initial Findings
- Wide-shallow architectures show better performance than deep-narrow ones
- Possible hypotheses:
  1. Enhanced attention capacity utilization in wider models
  2. Better gradient flow in shallower architectures
  3. More direct representation learning capabilities
  4. Smoother optimization landscape

## Installation
\`\`\`bash
# Clone the repository
git clone https://github.com/YourUsername/grokky.git
cd grokky

# Install dependencies
pip install -e .
\`\`\`

## Usage
\`\`\`bash
# Run a comparative study
python -m src.main --n-epochs 500 --batch-size 64

# With early stopping
python -m src.main --early-stopping --patience 20
\`\`\`
