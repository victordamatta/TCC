#!/bin/bash
# --freq_update 1 --keys_in_reply V --players "type=AI_NN,fs=50,args=backup/AI_SIMPLE|start/500|decay/0.99,fow=True;type=AI_SIMPLE,fs=20"
game=../rts/game_MC/game python3 train.py --batchsize 128 --players "type=AI_NN,fs=50;type=AI_SIMPLE,fs=50" --num_games 512 --tqdm --T 20 --additional_labels id,last_terminal  --agent_stats winrate "$@"

