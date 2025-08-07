import torch

def load_mimic_pth_into_actor_critic(model, mimic_pth_path):
    """
    Load mimic_model_V6_1.pth (PyTorch state_dict) into PPO's actor part (shared + actor).
    """
    mimic_state_dict = torch.load(mimic_pth_path, map_location='cpu')

    # load the corresponding layer
    with torch.no_grad():
        # === shared[0] = Linear(4 → 128)
        model.shared[0].weight.copy_(mimic_state_dict['net.0.weight'])
        model.shared[0].bias.copy_(mimic_state_dict['net.0.bias'])

        # actor[0] = Linear(128 → 64)
        model.actor[0].weight.copy_(mimic_state_dict['net.2.weight'])
        model.actor[0].bias.copy_(mimic_state_dict['net.2.bias'])

        # actor[2] = Linear(64 → 2)
        model.actor[2].weight.copy_(mimic_state_dict['net.4.weight'])
        model.actor[2].bias.copy_(mimic_state_dict['net.4.bias'])

    print("mimic_model_V6_1.pth successfully loaded into PPO actor!")
