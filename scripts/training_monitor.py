"""
Real-Time Training Monitor
===========================

Monitors training progress, alerts on issues, and enables early stopping.

Author: Evan Petersen
Date: November 2025
"""

import time
import torch
import json
from pathlib import Path
from datetime import datetime, timedelta
import threading


class TrainingMonitor:
    """Monitor training in real-time with alerts."""
    
    def __init__(self, 
                 log_file="training_monitor.log",
                 alert_on_nan=True,
                 alert_on_plateau=True,
                 alert_on_overfitting=True):
        
        self.log_file = Path(log_file)
        self.alert_on_nan = alert_on_nan
        self.alert_on_plateau = alert_on_plateau
        self.alert_on_overfitting = alert_on_overfitting
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_acc_top5': [],
            'lr': [],
            'epoch_time': []
        }
        
        self.start_time = None
        self.alerts = []
    
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self._log("="*80)
        self._log("TRAINING MONITOR STARTED")
        self._log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("="*80)
    
    def log_epoch(self, epoch, metrics):
        """Log epoch metrics and check for issues."""
        
        # Store metrics
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        # Check for NaN
        if self.alert_on_nan:
            for key, value in metrics.items():
                if isinstance(value, float) and (torch.isnan(torch.tensor(value)) or value != value):
                    self._alert(f"NaN detected in {key} at epoch {epoch}!")
        
        # Check for plateau (no improvement in 10 epochs)
        if self.alert_on_plateau and epoch >= 10:
            recent_val_acc = self.history['val_acc'][-10:]
            if max(recent_val_acc) - min(recent_val_acc) < 0.5:
                self._alert(f"Plateau detected: Val accuracy not improving (last 10 epochs)")
        
        # Check for overfitting
        if self.alert_on_overfitting and len(self.history['train_acc']) >= 5:
            train_acc = self.history['train_acc'][-1]
            val_acc = self.history['val_acc'][-1]
            gap = train_acc - val_acc
            
            if gap > 15.0:
                self._alert(f"Potential overfitting: Train-Val gap = {gap:.2f}%")
        
        # Log summary
        self._log(f"\nEpoch {epoch}:")
        for key, value in metrics.items():
            self._log(f"  {key}: {value:.4f}")
        
        # Estimated time remaining
        if len(self.history['epoch_time']) > 0:
            avg_epoch_time = sum(self.history['epoch_time']) / len(self.history['epoch_time'])
            remaining_epochs = metrics.get('total_epochs', 60) - epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            eta = str(timedelta(seconds=int(eta_seconds)))
            self._log(f"  ETA: {eta}")
    
    def should_stop_early(self, patience=15):
        """Check if training should stop early."""
        if len(self.history['val_acc']) < patience:
            return False
        
        best_idx = self.history['val_acc'].index(max(self.history['val_acc']))
        epochs_since_best = len(self.history['val_acc']) - best_idx - 1
        
        if epochs_since_best >= patience:
            self._alert(f"Early stopping: No improvement for {patience} epochs")
            return True
        
        return False
    
    def get_best_epoch(self):
        """Get epoch with best validation accuracy."""
        if not self.history['val_acc']:
            return None
        
        best_idx = self.history['val_acc'].index(max(self.history['val_acc']))
        return best_idx + 1, self.history['val_acc'][best_idx]
    
    def save_history(self, path="training_history_1000classes.json"):
        """Save training history to JSON."""
        output = {
            'history': self.history,
            'alerts': self.alerts,
            'total_time': time.time() - self.start_time if self.start_time else 0
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        self._log(f"\nHistory saved to: {path}")
    
    def _log(self, message):
        """Write to log file."""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        print(message)
    
    def _alert(self, message):
        """Log alert."""
        alert_msg = f"⚠️  ALERT: {message}"
        self.alerts.append({
            'time': datetime.now().isoformat(),
            'message': message
        })
        self._log("\n" + "="*80)
        self._log(alert_msg)
        self._log("="*80)


class CheckpointManager:
    """Manage model checkpoints intelligently."""
    
    def __init__(self, 
                 checkpoint_dir="data/checkpoints",
                 keep_best=3,
                 keep_every_n=10):
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.keep_every_n = keep_every_n
        self.checkpoints = []
    
    def save_checkpoint(self, 
                       model, 
                       optimizer, 
                       epoch, 
                       val_acc,
                       val_acc_top5=None,
                       is_best=False):
        """Save checkpoint with metadata."""
        
        # Filename
        if is_best:
            filename = "landmark_detector_1000classes_best.pth"
        else:
            filename = f"landmark_detector_1000classes_epoch_{epoch}.pth"
        
        filepath = self.checkpoint_dir / filename
        
        # Save
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_acc_top5': val_acc_top5,
            'timestamp': datetime.now().isoformat(),
            'num_classes': 1000
        }
        
        torch.save(checkpoint, filepath)
        
        self.checkpoints.append({
            'epoch': epoch,
            'val_acc': val_acc,
            'path': filepath,
            'is_best': is_best
        })
        
        print(f"✓ Checkpoint saved: {filepath}")
        
        # Cleanup old checkpoints
        if not is_best and epoch % self.keep_every_n != 0:
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only best N."""
        # Sort by validation accuracy
        sorted_ckpts = sorted(
            [c for c in self.checkpoints if not c['is_best']],
            key=lambda x: x['val_acc'],
            reverse=True
        )
        
        # Keep top N
        to_keep = set(c['path'] for c in sorted_ckpts[:self.keep_best])
        
        # Keep every Nth epoch
        to_keep.update(c['path'] for c in self.checkpoints 
                      if c['epoch'] % self.keep_every_n == 0)
        
        # Keep best
        to_keep.update(c['path'] for c in self.checkpoints if c['is_best'])
        
        # Delete others
        for ckpt in self.checkpoints:
            if ckpt['path'] not in to_keep and ckpt['path'].exists():
                try:
                    ckpt['path'].unlink()
                    print(f"  Cleaned up: {ckpt['path'].name}")
                except:
                    pass
