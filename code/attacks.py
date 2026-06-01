import torch


def L2_attack(input, model, label, criterion, epsilon_factor=0.1):
    """
    Generate L2-bounded adversarial examples using one-step gradient attack.
    
    Args:
        input: Input tensor
        model: Neural network model
        label: Target labels
        criterion: Loss function
        epsilon_factor: Controls perturbation magnitude (relative to input norm)
        
    Returns:
        adversarial: Adversarial examples
    """
    input = input.clone().detach().requires_grad_(True)
    output = model(input)
    loss = criterion(output, label)
    loss.backward()

    # Compute adaptive epsilon based on input norms
    norm = torch.linalg.norm(input, ord=2, dim=1)
    avg_norm = norm.mean()
    epsilon = epsilon_factor * avg_norm

    # Compute gradient direction and apply perturbation
    grad = input.grad
    grad_norm = torch.linalg.norm(grad, ord=2, dim=1, keepdim=True).clamp(min=1e-8)
    unit_direction = grad / grad_norm

    perturbation = epsilon * unit_direction
    adversarial = input + perturbation

    return adversarial.detach()
