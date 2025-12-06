# /// script
# dependencies = [
#     "torchvision",
#     "x-evolution>=0.0.20"
# ]
# ///

"""
Modified MNIST training script with fitness tracking and evaluation.
Adds logging similar to AudioMNIST training script.
"""

import torch
from torch import tensor, nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP
from x_mlps_pytorch.noisable import Noisable

# Monkeypatch Noisable to support int32 parameters
original_get_noised_params = Noisable.get_noised_params

def get_noised_params_patched(self, noise_for_params=dict(), inplace=False, noise_scale=None, negate=False, add_to_grad=False):
    # If adding to grad (optimizer step), we might need float gradients for int params? 
    # Actually, IntegerSGD expects float gradients (d_p = p.grad).
    # If p is int32, p.grad must be float32 usually for PyTorch. 
    # But if inplace and not add_to_grad (perturbation step), we are modifying p itself.
    
    # We will let the original method determine noise, but we intercept the final addition.
    # It's cleaner to copy the logic or subclass.
    # Let's copy the critical loop logic because we need to modify how noise is added.
    
    # get named params
    named_params = dict(self.model.named_parameters())

    if not inplace:
        from copy import deepcopy
        noised_params = deepcopy(named_params)
        return_params = noised_params
    else:
        return_params = named_params

    for name, param in named_params.items():
        param_shape = param.shape
        noise_or_seed = noise_for_params.get(name, None)
        
        # default helper function support from original file (we can't access them easily if private)
        # We can replicate 'default' logic:
        param_noise_scale = noise_scale if noise_scale is not None else self.noise_scale

        if noise_or_seed is None:
            continue

        # determine the noise
        if isinstance(noise_or_seed, int):
            from x_mlps_pytorch.noisable import with_seed
            noise = with_seed(noise_or_seed)(self.create_noise_fn)(param_shape)

        elif isinstance(noise_or_seed, tuple) and len(noise_or_seed) == 2:
            from x_mlps_pytorch.noisable import with_seed
            seed, noise_scale_with_seed = noise_or_seed
            noise = with_seed(seed)(self.create_noise_fn)(param_shape)

            if self.overridable_noise_scale:
                param_noise_scale = noise_scale_with_seed
            else:
                param_noise_scale *= noise_scale_with_seed

        elif torch.is_tensor(noise_or_seed):
            noise = noise_or_seed
        else:
            raise ValueError('invalid type, noise must be float tensor or int')

        noise = noise.to(self.device)

        # scale the noise
        if negate:
            param_noise_scale *= -1

        if param_noise_scale != 1.:
            noise = noise * param_noise_scale

        # --- MODIFICATION START ---
        # Handle int32 parameters by rounding noise
        if not param.is_floating_point() and not add_to_grad:
            # For perturbation, we round noise to integer before adding
            # This implements "stochastic rounding" effectively if noise is large enough?
            # Or just standard rounding.
            # If noise is float 0.4, rounded is 0. 
            # If we want stochastic, we should probably do: floor(noise + rand).
            # But here noise IS the random perturbation.
            # So if we simply round, we are perturbing by integers.
            noise = noise.round().to(param.dtype)
        # --- MODIFICATION END ---

        if inplace and not add_to_grad:
            param.data.add_(noise)

        elif inplace and add_to_grad:
            # adding noise inplace to grads
            if not param.is_floating_point():
                 # Handle int parameters: use custom float_grad attribute
                 if not hasattr(param, 'float_grad'):
                     param.float_grad = noise.clone()
                 else:
                     param.float_grad.add_(noise)
            else:
                if param.grad is None:
                    param.grad = noise.clone()
                else:
                    param.grad.add_(noise)
        else:
            noised_params[name] = param + noise

    return return_params

Noisable.get_noised_params = get_noised_params_patched
from torch.utils.data import DataLoader
from pathlib import Path
import csv
import sys
import re
from io import StringIO

# model

# integer model

class IntegerSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                d_p = None
                if hasattr(p, 'float_grad'):
                     d_p = p.float_grad
                elif p.grad is not None:
                     d_p = p.grad
                
                if d_p is None:
                    continue
                
                # Gradient from ES is float
                
                # Update typically: p -= lr * grad
                # We want stochastic rounding:
                # delta = -lr * grad
                # p_new = p + floor(delta + rand)
                
                delta = -lr * d_p
                
                # Stochastic rounding
                # Add uniform noise [0, 1) then floor
                noise = torch.rand_like(delta)
                update_int = torch.floor(delta + noise)
                
                # Apply update. Check if param is integer type
                if p.dtype.is_floating_point:
                    p.data.add_(update_int)
                else:
                    p.data.add_(update_int.to(p.dtype))

        return loss

    def zero_grad(self, set_to_none=False):
        super().zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'float_grad'):
                    if set_to_none:
                        p.float_grad = None
                    else:
                        p.float_grad.zero_()

class IntegerLinear(nn.Module):
    def __init__(self, dim_in, dim_out, scale=1.0):
        super().__init__()
        # Strict integer storage
        self.weight = nn.Parameter(torch.zeros(dim_out, dim_in, dtype=torch.int32), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(dim_out, dtype=torch.int32), requires_grad=False)
        self.scale = scale
        
        # Initialize with integer values
        # We need temporary float for initialization helpers, then cast
        with torch.no_grad():
            w_init = torch.empty_like(self.weight, dtype=torch.float32)
            nn.init.uniform_(w_init, -2.4, 2.4)
            self.weight.data.copy_(w_init.round().int())
            
            b_init = torch.empty_like(self.bias, dtype=torch.float32)
            nn.init.uniform_(b_init, -0.4, 0.4)
            self.bias.data.copy_(b_init.round().int())

    def forward(self, x):
        # Cast to float for computation (F.linear doesn't support int32 inputs/weights usually)
        w = self.weight.float()
        b = self.bias.float()
        
        out = F.linear(x, w, b)
        
        if self.scale != 1.0:
            out = (out / self.scale)
            
        # Keep activations integer
        return out.floor()

model = nn.Sequential(
    nn.Flatten(),
    # Layer 1: Inputs 0-255. Input variance ~ 255^2.
    # Divide by 255 to roughly normalize?
    # Or divide by sqrt(784) * something?
    # Let's try aggressive scaling to keep activations small integers.
    IntegerLinear(784, 128, scale=10.0), 
    nn.ReLU(),
    IntegerLinear(128, 10, scale=10.0)
)
# data

train_dataset = datasets.MNIST('./data', train = True, download = True, transform = transforms.ToTensor())
test_dataset = datasets.MNIST('./data', train = False, download = True, transform = transforms.ToTensor())

# fitness as inverse of loss

class MnistFitness:
    def __init__(self, dataset, device, batch_size=512):
        self.batch_size = batch_size
        self.device = device
        
        # Preload data to device
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data, target = next(iter(loader))
        self.data = data.to(device)
        self.target = target.to(device)
        
    def __call__(self, model):
        # Fast random sampling
        indices = torch.randint(0, len(self.data), (self.batch_size,), device=self.device)
        
        batch_data = self.data[indices]
        batch_target = self.target[indices]

        # Pass "integers" (stored as float for compatibility, but values are 0-255)
        # We need to ensure input is float dtype for F.linear but holds integer values
        batch_data = batch_data.float()

        with torch.inference_mode():
            logits = model(batch_data)
            loss = F.cross_entropy(logits, batch_target)

        return -loss


def evaluate_model(model, dataset, device, batch_size=128):
    """Evaluate model accuracy on a dataset."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            # Match training distribution: inputs 0-255
            data = data * 255.0
            
            logits = model(data)
            loss = F.cross_entropy(logits, target)

            total_loss += loss.item() * data.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return accuracy, avg_loss


class StdoutCapture:
    """Capture stdout and parse fitness values from x-evolution output."""

    def __init__(self):
        self.fitness_history = []
        self.original_stdout = sys.stdout
        self.capture_buffer = StringIO()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.original_stdout

    def write(self, text):
        # Write to both original stdout and capture buffer
        self.original_stdout.write(text)
        self.original_stdout.flush()

        # Parse fitness from x-evolution output
        # Format: "[1000] average fitness: -124.247 | fitness std: 8.883"
        match = re.search(r'\[(\d+)\]\s+average fitness:\s+([-\d.]+)', text)
        if match:
            generation = int(match.group(1))
            avg_fitness = float(match.group(2))
            self.fitness_history.append({
                'generation': generation,
                'avg_fitness': avg_fitness
            })

    def flush(self):
        self.original_stdout.flush()


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    # Move model to device
    model.to(DEVICE)

    # Compile model for faster execution
    print("Compiling model with torch.compile...")
    compiled_model = torch.compile(model)

    print(f"Model Parameters: {sum(p.numel() for p in compiled_model.parameters()):,}")
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Test dataset size: {len(test_dataset)} samples")

    # evo
    from x_evolution import EvoStrategy

    # Initialize fitness function with cached data
    loss_mnist = MnistFitness(train_dataset, DEVICE)

    evo_strat = EvoStrategy(
        compiled_model,
        environment = loss_mnist,
        noise_population_size = 100,
        noise_scale = 1.0,
        noise_low_rank = 2,
        num_generations = 1000,
        learning_rate = 0.1,
        optimizer_klass = IntegerSGD,
        optimizer_kwargs = {}  # No specific kwargs needed for IntegerSGD defaults
    )

    print("\nStarting Evolutionary Training...")
    print("=" * 60)

    # Train and capture fitness history from printed output
    with StdoutCapture() as capture:
        evo_strat()

    fitness_history = capture.fitness_history

    print("\nTraining complete!")
    print(f"Captured {len(fitness_history)} fitness values")

    # Save fitness history to CSV
    checkpoint_dir = Path('./checkpoints_mnist')
    checkpoint_dir.mkdir(exist_ok=True)

    if fitness_history:
        csv_path = checkpoint_dir / 'fitness_history_no_WD.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['generation', 'avg_fitness'])
            writer.writeheader()
            writer.writerows(fitness_history)
        print(f"Saved fitness history: {csv_path}")

    final_checkpoint = checkpoint_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'fitness_history': fitness_history,
    }, final_checkpoint)
    print(f"Saved final model: {final_checkpoint}")

    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    train_acc, train_loss = evaluate_model(model, train_dataset, DEVICE)
    print(f"Train - Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")

    test_acc, test_loss = evaluate_model(model, test_dataset, DEVICE)
    print(f"Test  - Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
