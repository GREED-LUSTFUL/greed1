#!/usr/bin/env bash
set -e

# Direct link to your HF‑hosted model
MODEL_URL="https://huggingface.co/greed36/intelliflight-delay-model/resolve/main/flight_delay_model.pkl"
TARGET="flight_delay_model.pkl"

if [ ! -f "$TARGET" ]; then
  echo "⬇️  Downloading model from Hugging Face…"
  curl -L "$MODEL_URL" -o "$TARGET"
  echo "✅  Model saved to $TARGET"
fi
