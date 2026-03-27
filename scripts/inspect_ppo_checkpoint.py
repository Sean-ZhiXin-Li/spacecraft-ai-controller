import torch


def main():
    ckpt = torch.load("ppo_orbit/ppo_best_model.pth", map_location="cpu")

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        print("Using checkpoint['model_state']")
    else:
        state_dict = ckpt
        print("Using checkpoint as state_dict")

    print("-" * 60)
    for key, value in state_dict.items():
        print(f"{key:30s} {tuple(value.shape)}")


if __name__ == "__main__":
    main()