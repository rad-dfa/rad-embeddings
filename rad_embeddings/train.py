from encoder import EncoderModule

EncoderModule.train(max_size=10, n_tokens=10, debug=True, save_dir="storage", binary_reward=True, enable_wandb=True)