# Usage

Below is a quick example of using our codebase for training CNN models with mixed ghost clipping:

```python
import torchvision, torch, opacus
from private_cnns import PrivacyEngine

model = torchvision.models.resnet18()

# replace BatchNorm by GroupNorm or LayerNorm
model=opacus.validators.ModuleValidator.fix(model)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
privacy_engine = PrivacyEngine(
    model,
    batch_size=256,
    sample_size=50000,
    epochs=3,
    max_grad_norm=0.1,
    target_epsilon=3,
    mode='ghost-mixed',  # use `mode='ghost'` for ghost clipping
)
privacy_engine.attach(optimizer)

# Same training procedure, e.g. data loading, forward pass, logits...
loss = F.cross_entropy(logits, labels, reduction="none")
# do not use loss.backward()
optimizer.step(loss=loss)
```

A special use of our privacy engine is to use the gradient accumulation. This is achieved with virtual step function.

```python
import torchvision, torch, timm
from private_cnns import PrivacyEngine

gradient_accumulation_steps = 10  

# Batch size/physical batch size. Take an update once this many iterations

model = torchvision.models.resnet18()
model=opacus.validators.ModuleValidator.fix(model)
optimizer = torch.optim.Adam(model.parameters())
privacy_engine = PrivacyEngine(...)
privacy_engine.attach(optimizer)

for i, batch in enumerate(dataloader):
    loss = model(batch)
    if i % gradient_accumulation_steps == 0:
        optimizer.step(loss=loss)
        optimizer.zero_grad()
    else:
        optimizer.virtual_step(loss=loss)
```
