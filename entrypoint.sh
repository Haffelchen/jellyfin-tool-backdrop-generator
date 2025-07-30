#!/bin/bash

DEFAULT_LENGTH=60
DEFAULT_RESOLUTION=1080
DEFAULT_CRF=28
DEFAULT_TIMEOUT=60
DEFAULT_INTERVAL=21600

LENGTH_VAL="${LENGTH:-$DEFAULT_LENGTH}"
RESOLUTION_VAL="${RESOLUTION:-$DEFAULT_RESOLUTION}"
CRF_VAL="${CRF:-$DEFAULT_CRF}"
TIMEOUT_VAL="${TIMEOUT:-$DEFAULT_TIMEOUT}"
INTERVAL_VAL="${INTERVAL:-$DEFAULT_INTERVAL}"

echo "Launching backdrop generator with:"
echo "  LENGTH $LENGTH_VAL"
echo "  RESOLUTION $RESOLUTION_VAL"
echo "  CRF $CRF_VAL"
echo "  TIMEOUT $TIMEOUT_VAL"
echo "  INTERVAL $INTERVAL_VAL"
echo "  FORCE=$FORCE"
echo "  DAEMON=$DAEMON"
echo "  NO_AUDIO=$NO_AUDIO"
echo "  FFMPEG_EXTRA=$FFMPEG_EXTRA"

args=(
  --movies /movies
  --tv /tv
  --length "$LENGTH_VAL"
  --resolution "$RESOLUTION_VAL"
  --crf "$CRF_VAL"
  --timeout "$TIMEOUT_VAL"
  --interval "$INTERVAL_VAL"
)

[ "$FORCE" = "true" ] && args+=(--force)
[ "$DAEMON" = "true" ] && args+=(--daemon)
[ "$NO_AUDIO" = "true" ] && args+=(--no-audio)
[ -n "$FFMPEG_EXTRA" ] && args+=(--ffmpeg-extra "$FFMPEG_EXTRA")

exec python media_theme_processor.py "${args[@]}"