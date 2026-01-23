#!/bin/bash
# Quick-start script for running Chichewa HarmBench evaluation
# Author: Wassim Hamidouche
# Date: January 9, 2026

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   HarmBench Chichewa Evaluation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if model is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: $0 <model_name_or_path> [task_name]${NC}"
    echo ""
    echo "Examples:"
    echo "  $0 meta-llama/Llama-2-7b-chat-hf"
    echo "  $0 meta-llama/Llama-2-7b-chat-hf harmbench_direct_request_chichewa"
    echo "  $0 /path/to/local/model harmbench_chichewa"
    echo ""
    echo "Available Chichewa tasks:"
    echo "  - harmbench_chichewa                      (both tasks)"
    echo "  - harmbench_direct_request_chichewa       (DirectRequest only)"
    echo "  - harmbench_human_jailbreaks_chichewa     (HumanJailbreaks only)"
    echo ""
    exit 1
fi

MODEL=$1
TASK=${2:-harmbench_chichewa}  # Default to both tasks

echo -e "${GREEN}Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Task:  $TASK"
echo ""

# Check GPT-5 credentials
echo -e "${YELLOW}Checking GPT-5 credentials...${NC}"
if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  Warning: AZURE_OPENAI_API_KEY not set${NC}"
    echo "  Evaluation requires GPT-5 API for harm classification"
    echo "  See QUICK_START_GPT5.md for setup instructions"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Azure OpenAI credentials found${NC}"
fi
echo ""

# Navigate to evaluation directory
EVAL_DIR="/home/whamidouche/ssdprivate/git-local/low-resource-llm.pipeline/multilingual-llm-evaluation"
cd "$EVAL_DIR"

echo -e "${BLUE}Starting evaluation...${NC}"
echo ""

# Run evaluation
lm_eval --model hf \
    --model_args pretrained="$MODEL" \
    --tasks "$TASK" \
    --device cuda:0 \
    --batch_size auto \
    --output_path "./results/harmbench_chichewa_$(date +%Y%m%d_%H%M%S)" \
    --log_samples

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   Evaluation Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Results saved to: ./results/"
echo ""
echo "To run English + Chichewa comparison:"
echo "  $0 $MODEL harmbench,harmbench_chichewa"
