import os
from rad_embeddings import EncoderModule, DynEncoderModule

os.makedirs("rudy_storage", exist_ok=True)

# baseline
# EncoderModule.train(
#     seed=42,
#     max_size=10,
#     n_tokens=5,
#     enable_wandb=True,
#     wandb_entity="rcolato29-university-of-california-berkeley",
#     save_dir="rudy_storage",
#     log="rudy_storage/log.csv",
# )

# dynamic alphabet
DynEncoderModule.train(
    seed=42,
    max_size=10,
    n_tokens=5,
    n_events=5,
    total_timesteps=2e6,
    enable_wandb=True,
    wandb_entity="rcolato29-university-of-california-berkeley",
    save_dir="rudy_storage",
    log="rudy_storage/log.csv",
)