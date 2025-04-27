import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small
from config import TrainingConfig
from utils import create_dataloaders, evaluate, plot_training_history
import os
from tqdm import tqdm

try:
    from perforatedai import pb_globals as PBG
    from perforatedai import pb_utils as PBU
    PAI_AVAILABLE = True
except ImportError:
    PAI_AVAILABLE = False

def setup_output_dirs(model_name):
    """Create output directories for logs and images"""
    base_dir = os.path.join(os.getcwd(), model_name)
    log_dir = os.path.join(base_dir, 'logs')
    image_dir = os.path.join(base_dir, 'images')
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    return log_dir, image_dir

def setup_pai_model(model, config):
    PBG.switchMode = config.pai_config["switch_mode"]
    PBG.nEpochsToSwitch = config.pai_config["n_epochs_to_switch"]
    PBG.capAtN = config.pai_config["cap_at_n"]
    PBG.pEpochsToSwitch = config.pai_config["p_epochs_to_switch"]
    PBG.inputDimensions = config.pai_config["input_dimensions"]
    PBG.historyLookback = config.pai_config["history_lookback"]
    PBG.maxDendrites = config.pai_config["max_dendrites"]
    PBG.unwrappedModulesConfirmed = False
    PBG.testingDendriteCapacity = False
    PBG.moduleNamesToConvert += config.pai_config["modules_to_convert"]
    
    model = PBU.initializePB(model)
    PBG.pbTracker.setOptimizer(torch.optim.Adam)
    PBG.pbTracker.setScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    return model

def train_model(config):
    log_dir, image_dir = setup_output_dirs(config.model_name)
    
    train_loader, val_loader, test_loader = create_dataloaders(config.batch_size)
    model = mobilenet_v3_small(width_mult=config.width_mult).to(config.device)
    
    if config.use_pai:
        if not PAI_AVAILABLE:
            raise ImportError("PerforatedAI package not found but use_pai=True")
        model = setup_pai_model(model, config)
        
    criterion = nn.CrossEntropyLoss()
    
    if config.use_pai:
        optimizerArgs = {'params': model.parameters(), 'lr': config.learning_rate}
        schedulerArgs = {'mode': 'max', 'patience': 5}
        optimizer, scheduler = PBG.pbTracker.setupOptimizer(model, optimizerArgs, schedulerArgs)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5
        )
    
    best_val_acc = 0
    best_model_state = None
    train_accuracies = []
    val_accuracies = []
    epochs = []
    
    pbar = tqdm(range(config.num_epochs), desc="Training")
    for epoch in pbar:
        model.train()
        correct = 0
        total_loss = 0
        
        train_pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}")
        for images, labels in train_pbar:
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        train_accuracy = correct / len(train_loader.dataset)
        train_accuracies.append(train_accuracy)
        epochs.append(epoch + 1)
        
        # Validation
        model.eval()
        correct = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
        
        val_accuracy = correct / len(val_loader.dataset)
        val_accuracies.append(val_accuracy)
        
        avg_loss = total_loss/len(train_loader)
        pbar.set_description(f"Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        
        # Log every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            print(f"Training Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Only step scheduler in non-PAI mode
        if not config.use_pai:
            scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"\nNew best model saved with validation accuracy: {val_accuracy:.4f}")
    
    model_type = "pai" if config.use_pai else "plain"
    model_save_path = os.path.join(log_dir, f"best_model_{model_type}.pth")
    history_plot_path = os.path.join(image_dir, f"training_history_{model_type}.png")
    
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, model_save_path)
    
    plot_training_history(
        epochs, 
        train_accuracies, 
        val_accuracies, 
        history_plot_path
    )
    
    test_accuracy = evaluate(model, test_loader, config.device, return_accuracy=True)
    log_path = os.path.join(log_dir, f"training_log_{model_type}.txt")

    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {model_save_path}")
    print(f"Training history plot saved to: {history_plot_path}")
    print(f"Training log saved to: {log_path}")

    with open(log_path, 'w') as f:
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Type: {model_type}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Width Multiplier: {config.width_mult}\n")
        f.write(f"Learning Rate: {config.learning_rate}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Final Test Accuracy: {test_accuracy:.4f}\n")
    
    return model, test_accuracy

def train_both_models(config):
    """
    Trains both the standard (plain) and PerforatedAI (PAI) versions of a model using the provided configuration.

    Args:
        config (TrainingConfig): The configuration object containing training parameters. The function will toggle the 'use_pai' attribute to train both variants.

    Returns:
        tuple: (plain_model, plain_accuracy, pai_model, pai_accuracy)
            - plain_model: The trained standard model.
            - plain_accuracy: Test accuracy of the standard model.
            - pai_model: The trained PAI model.
            - pai_accuracy: Test accuracy of the PAI model.
    """
    print(f"\nTraining plain model for {config.model_name}...")
    config.use_pai = False
    plain_model, plain_accuracy = train_model(config)
    
    print(f"\nTraining PAI model for {config.model_name}...")
    config.use_pai = True
    pai_model, pai_accuracy = train_model(config)
    
    print("\nModel Comparison:")
    print(f"Plain Model Accuracy: {plain_accuracy:.4f}")
    print(f"PAI Model Accuracy: {pai_accuracy:.4f}")
    print(f"Accuracy Difference (PAI - Plain): {(pai_accuracy - plain_accuracy):.4f}")
    
    return plain_model, plain_accuracy, pai_model, pai_accuracy

if __name__ == "__main__":
    # Configuration for MobileNet Small (Base version)
    small_config = TrainingConfig(
        batch_size=16,
        width_mult=1.0,
        learning_rate=0.001,
        model_name="mobilenet_small",
        use_pai=False,
        num_epochs=2
    )
    
    # Configuration for Extra Small V2
    v2_config = TrainingConfig(
        batch_size=16,
        width_mult=0.5,
        learning_rate=0.001,
        model_name="mobilenet_extra_small_v2",
        use_pai=False,
        num_epochs=150
    )
    
    # Configuration for Extra Small V3
    v3_config = TrainingConfig(
        batch_size=16,
        width_mult=0.75,
        learning_rate=0.001,
        model_name="mobilenet_extra_small_v3",
        use_pai=False,
        num_epochs=150
    )
    
    # Configuration for MobileNetV3 Large
    large_config = TrainingConfig(
        batch_size=16,
        width_mult=1.0,
        learning_rate=0.001,
        model_name="mobilenet_large",
        use_pai=False,
        num_epochs=170
    )
    
    print("\nTraining MobileNet Extra Small V2...")
    # _, _, _, _ = train_both_models(v2_config)

    print(f"\nTraining just PAI model for {v2_config.model_name}...")
    v2_config.use_pai = True
    _, _ = train_model(v2_config)
    
    # print("\nTraining Extra Small V2...")
    # plain_model_v2, plain_acc_v2, pai_model_v2, pai_acc_v2 = train_both_models(v2_config)
    
    # print("\nTraining Extra Small V3...")
    # plain_model_v3, plain_acc_v3, pai_model_v3, pai_acc_v3 = train_both_models(v3_config)
    
    # print("\nTraining MobileNet Large...")
    # plain_model_large, plain_acc_large, pai_model_large, pai_acc_large = train_both_models(large_config)
    
    # print("\nFinal Comparison of All Models:")
    # print("\nMobileNetV3 Small:")
    # print(f"Plain: {plain_acc_small:.4f}, PAI: {pai_acc_small:.4f}, Diff: {(pai_acc_small - plain_acc_small):.4f}")
    
    # print("\nExtra Small V2:")
    # print(f"Plain: {plain_acc_v2:.4f}, PAI: {pai_acc_v2:.4f}, Diff: {(pai_acc_v2 - plain_acc_v2):.4f}")
    
    # print("\nExtra Small V3:")
    # print(f"Plain: {plain_acc_v3:.4f}, PAI: {pai_acc_v3:.4f}, Diff: {(pai_acc_v3 - plain_acc_v3):.4f}")
    
    # print("\nMobileNet Large:")
    # print(f"Plain: {plain_acc_large:.4f}, PAI: {pai_acc_large:.4f}, Diff: {(pai_acc_large - plain_acc_large):.4f}")