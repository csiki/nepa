# Intermediate Loss Implementation Plan

## Goal

Add a training mode where NEPA loss is applied at **every intermediate ViT layer**, not just the final output. This enables comparing convergence speed between:
- **Baseline**: Final-layer NEPA loss only
- **Deep supervision**: NEPA loss at every layer

## Wandb Logging

Three separate losses logged:
- `loss` — combined total (used for backprop)
- `loss_final` — original end-layer loss
- `loss_intermediate` — mean of all intermediate layer losses

---

## Files to Modify

### 1. `models/vit_nepa/configuration_vit_nepa.py`

Add config parameter:

```python
use_intermediate_loss: bool = False
```

**Location**: Add to `__init__` parameters (around line 134) and store as `self.use_intermediate_loss`.

---

### 2. `models/vit_nepa/modeling_vit_nepa.py`

#### 2a. Update `EmbeddedModelingOutput` (lines 221-260)

Add optional fields for separate loss components:

```python
loss_final: Optional[torch.FloatTensor] = None
loss_intermediate: Optional[torch.FloatTensor] = None
```

#### 2b. Modify `ViTNepaForPreTraining.forward()` (lines 851-883)

Current logic:
```python
outputs = self.vit_nepa(...)
embedded_loss = prediction_loss(sequence_input, sequence_output)
return EmbeddedModelingOutput(loss=embedded_loss, ...)
```

New logic:
```python
# Force hidden states output when intermediate loss is enabled
use_intermediate = self.config.use_intermediate_loss

outputs = self.vit_nepa(
    ...,
    output_hidden_states=use_intermediate,  # add this
)

sequence_input = outputs.input_embedding
sequence_output = outputs.last_hidden_state

# Final layer loss (always computed)
loss_final = prediction_loss(sequence_input, sequence_output)

# Intermediate losses (only when enabled)
loss_intermediate = None
if use_intermediate and outputs.hidden_states is not None:
    intermediate_losses = []
    # hidden_states[0] is embedding, hidden_states[1:] are layer outputs
    for layer_hidden in outputs.hidden_states[1:-1]:  # skip input embedding and final
        layer_loss = prediction_loss(sequence_input, layer_hidden)
        intermediate_losses.append(layer_loss)
    if intermediate_losses:
        loss_intermediate = torch.stack(intermediate_losses).mean()

# Combined loss
if loss_intermediate is not None:
    total_loss = loss_final + loss_intermediate
else:
    total_loss = loss_final

return EmbeddedModelingOutput(
    loss=total_loss,
    loss_final=loss_final,
    loss_intermediate=loss_intermediate,
    hidden_states=outputs.hidden_states,
    attentions=outputs.attentions,
)
```

---

### 3. `run_nepa.py`

#### 3a. Add CLI argument in `ModelArguments` (around line 285)

```python
use_intermediate_loss: bool = field(
    default=False,
    metadata={"help": "Apply NEPA loss at every intermediate layer."},
)
```

#### 3b. Pass to config (around line 471)

After loading config, set the flag:
```python
config.use_intermediate_loss = model_args.use_intermediate_loss
```

#### 3c. Log extra losses in `EnhancedTrainer`

Override `compute_loss` or add custom logging. Simplest approach — override `training_step` to log after computing loss:

```python
def training_step(self, model, inputs, num_items_in_batch=None):
    loss = super().training_step(model, inputs, num_items_in_batch)
    
    # Get last forward outputs for logging
    if hasattr(model, '_last_outputs'):
        outputs = model._last_outputs
        if outputs.loss_final is not None:
            self.log({"loss_final": outputs.loss_final.detach().item()})
        if outputs.loss_intermediate is not None:
            self.log({"loss_intermediate": outputs.loss_intermediate.detach().item()})
    
    return loss
```

Alternative (cleaner): override `compute_loss` and use `self.state` to store extra losses, then log in `_maybe_log_save_evaluate`.

**Simplest approach**: Just rely on HF Trainer's built-in behavior. The Trainer logs all scalar values from the output dict. We can modify `EmbeddedModelingOutput` to return a dict-like structure, or simply add the extra losses to the model output and let them be logged automatically if we return them as part of a dict.

Actually, the cleanest way: In `compute_loss`, extract extra losses and store them, then log them. Here's the implementation:

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    outputs = model(**inputs)
    loss = outputs.loss
    
    # Store extra losses for logging
    if outputs.loss_final is not None:
        self._extra_losses = {
            "loss_final": outputs.loss_final.detach(),
            "loss_intermediate": outputs.loss_intermediate.detach() if outputs.loss_intermediate is not None else None,
        }
    
    return (loss, outputs) if return_outputs else loss

def log(self, logs, start_time=None):
    # Inject extra losses into logs
    if hasattr(self, '_extra_losses') and self._extra_losses:
        for k, v in self._extra_losses.items():
            if v is not None:
                logs[k] = v.item() if torch.is_tensor(v) else v
        self._extra_losses = {}
    super().log(logs, start_time)
```

---

## Usage

### Baseline training (no changes to script):
```bash
bash scripts/pretrain/nepa_b.sh
```

### Deep supervision training:
Add to script or command line:
```bash
--use_intermediate_loss True
```

---

## Summary of Changes

| File | Lines | Change |
|------|-------|--------|
| `configuration_vit_nepa.py` | ~134 | Add `use_intermediate_loss` param |
| `modeling_vit_nepa.py` | ~221 | Add `loss_final`, `loss_intermediate` to output class |
| `modeling_vit_nepa.py` | ~851-883 | Compute intermediate losses in `forward()` |
| `run_nepa.py` | ~285 | Add CLI arg `use_intermediate_loss` |
| `run_nepa.py` | ~471 | Pass flag to config |
| `run_nepa.py` | ~79 | Add logging for extra losses in `EnhancedTrainer` |
