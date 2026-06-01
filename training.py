from attacks import L2_attack


def clean_train(dataloader, criterion, optimizer, model, device):
    '''
    Standard training on clean data.
    
    Args:
        dataloader: PyTorch DataLoader with (x, y) pairs
        criterion: Loss function
        optimizer: Optimizer
        model: Neural network model
        device: torch.device (cpu or cuda)'''
    
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def adv_train(dataloader, criterion, optimizer, model, device, epsilon_factor=0.1):
    
    '''Adversarial training on mixture of clean and adversarial examples.
    Loss = 0.5 * L_clean + 0.5 * L_adv
    
    Args:
        dataloader: PyTorch DataLoader with (x, y) pairs
        criterion: Loss function
        optimizer: Optimizer
        model: Neural network model
        device: torch.device (cpu or cuda)
        epsilon_factor: Controls adversarial perturbation magnitude'''
    
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        # Generate adversarial examples
        attack = L2_attack(x, model, y, criterion, epsilon_factor=epsilon_factor)
        
        # Compute losses
        preds_adv = model(attack)
        preds_clean = model(x)
        loss_adv = criterion(preds_adv, y)
        loss_clean = criterion(preds_clean, y)
        
        # Mixed loss (50/50 clean and adversarial)
        loss_fn = 0.5 * loss_adv + 0.5 * loss_clean

        optimizer.zero_grad()
        loss_fn.backward()
        optimizer.step()
