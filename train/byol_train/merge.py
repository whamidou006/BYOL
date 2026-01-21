"""Model merging utilities for BYOL Training."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger("byol-train")


@dataclass
class MergeConfig:
    """Configuration for LoRA merging."""
    model_name_or_path: str = ""
    adapter_name_or_path: str = ""
    template: str = "gemma"
    export_dir: str = ""
    export_size: int = 2
    export_device: str = "auto"
    export_legacy_format: bool = False
    
    @classmethod
    def from_yaml(cls, path: Path) -> "MergeConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            model_name_or_path=data.get("model_name_or_path", ""),
            adapter_name_or_path=data.get("adapter_name_or_path", ""),
            template=data.get("template", "gemma"),
            export_dir=data.get("export_dir", ""),
            export_size=data.get("export_size", 2),
            export_device=data.get("export_device", "auto"),
            export_legacy_format=data.get("export_legacy_format", False),
        )
    
    def to_dict(self) -> Dict:
        """Convert to dict for LlamaFactory."""
        return {
            "model_name_or_path": self.model_name_or_path,
            "adapter_name_or_path": self.adapter_name_or_path,
            "template": self.template,
            "export_dir": self.export_dir,
            "export_size": self.export_size,
            "export_device": self.export_device,
            "export_legacy_format": self.export_legacy_format,
        }


def merge_lora(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    template: str = "gemma",
    export_size: int = 2,
    dry_run: bool = False,
) -> bool:
    """Merge LoRA adapter into base model.
    
    Args:
        base_model: Path to base model
        adapter_path: Path to LoRA adapter
        output_dir: Output directory for merged model
        template: Chat template name
        export_size: Number of shards
        dry_run: Print config without running
        
    Returns:
        True if successful, False otherwise
    """
    import tempfile
    
    config = MergeConfig(
        model_name_or_path=base_model,
        adapter_name_or_path=adapter_path,
        template=template,
        export_dir=output_dir,
        export_size=export_size,
    )
    
    logger.info("=" * 60)
    logger.info("🔀 Merging LoRA adapter")
    logger.info(f"   Base model: {base_model}")
    logger.info(f"   Adapter: {adapter_path}")
    logger.info(f"   Output: {output_dir}")
    logger.info("=" * 60)
    
    # Write temp config
    fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="merge_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    if dry_run:
        logger.info("[DRY RUN] Config:")
        logger.info(Path(temp_path).read_text())
        Path(temp_path).unlink()
        return True
    
    try:
        cmd = ["llamafactory-cli", "export", temp_path]
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            logger.info(f"✅ Merged model saved to: {output_dir}")
            return True
        else:
            logger.error(f"❌ Merge failed with exit code {result.returncode}")
            return False
    
    except Exception as e:
        logger.exception(f"Error: {e}")
        return False
    
    finally:
        Path(temp_path).unlink(missing_ok=True)
