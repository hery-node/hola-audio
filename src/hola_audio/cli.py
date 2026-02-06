"""Command-line interface for Hola Audio.

Usage:
    hola-audio start          Start the voice input service
    hola-audio transcribe     Transcribe an audio file
    hola-audio correct        Correct text via LLM
    hola-audio finetune       Manage fine-tuning workflow
    hola-audio devices        List audio input devices
    hola-audio config         Show/edit configuration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _cmd_start(args: argparse.Namespace) -> None:
    """Start the voice input service."""
    from hola_audio.app import Application
    from hola_audio.config import Config

    config = Config(args.config) if args.config else Config()
    app = Application(config)

    try:
        app.start(
            load_model=not args.no_model,
            enable_tray=not args.no_tray,
        )
        print("Hola Audio is running. Press Ctrl+C to stop.")
        print(f"  Toggle recording: {config.get('hotkey.toggle_recording')}")
        print(f"  Cancel recording: {config.get('hotkey.cancel_recording')}")
        print(f"  Correct clipboard: {config.get('hotkey.correct_clipboard')}")
        app.wait()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()


def _cmd_transcribe(args: argparse.Namespace) -> None:
    """Transcribe an audio file."""
    from hola_audio.app import Application
    from hola_audio.config import Config

    config = Config(args.config) if args.config else Config()
    app = Application(config)

    # Load model synchronously for CLI use
    app.engine.load(checkpoint_path=args.checkpoint)

    for audio_path in args.files:
        path = Path(audio_path)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            continue

        text = app.transcribe_file(path, correct=args.correct)
        if args.json:
            print(json.dumps({"file": str(path), "text": text}))
        else:
            if len(args.files) > 1:
                print(f"\n--- {path.name} ---")
            print(text)


def _cmd_correct(args: argparse.Namespace) -> None:
    """Correct text via LLM."""
    from hola_audio.config import Config
    from hola_audio.correction.client import CorrectionClient

    config = Config(args.config) if args.config else Config()
    client = CorrectionClient(
        endpoint=config.get("correction.endpoint", ""),
        api_key=config.get("correction.api_key", ""),
        model=config.get("correction.model", "gemini-2.0-flash"),
    )

    if not client.is_configured:
        print("Error: Correction endpoint not configured.", file=sys.stderr)
        print("Set 'correction.endpoint' in config or HOLA_AUDIO__CORRECTION__ENDPOINT env var.", file=sys.stderr)
        sys.exit(1)

    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        print("Reading from stdin (Ctrl+D to finish)...")
        text = sys.stdin.read()

    corrected = client.correct(text, context=args.context or "")
    print(corrected)


def _cmd_finetune(args: argparse.Namespace) -> None:
    """Manage fine-tuning workflow."""
    from hola_audio.audio.capture import AudioCapture
    from hola_audio.config import Config
    from hola_audio.finetune.dataset import FinetuneDataset
    from hola_audio.finetune.trainer import OfflineTrainer, OnlineTrainer

    config = Config(args.config) if args.config else Config()
    dataset = FinetuneDataset(config.finetune_dir)

    if args.action == "init":
        sentences = None
        if args.sentences_file:
            with open(args.sentences_file, "r", encoding="utf-8") as fh:
                sentences = [line.strip() for line in fh if line.strip()]
        dataset.initialize(sentences)
        print(f"Dataset initialized with {dataset.total} sentences")
        print(f"Data directory: {config.finetune_dir}")

    elif args.action == "status":
        print(f"Dataset: {dataset.total} sentences, {dataset.recorded_count} recorded")
        print(f"Progress: {dataset.progress:.0%}")
        if not dataset.is_complete:
            next_sample = dataset.get_next_unrecorded()
            if next_sample:
                print(f"\nNext sentence to record:")
                print(f"  [{next_sample.id}] {next_sample.text}")

    elif args.action == "record":
        capture = AudioCapture(
            sample_rate=config.get("audio.sample_rate", 16000),
            channels=config.get("audio.channels", 1),
            output_dir=dataset.audio_dir,
        )

        sample = dataset.get_next_unrecorded()
        if sample is None:
            print("All sentences have been recorded!")
            return

        print(f"\nPlease read the following sentence aloud:")
        print(f"  \"{sample.text}\"")
        print(f"\nPress Enter to start recording, then Enter again to stop...")
        input()

        capture.start()
        print("  Recording... (press Enter to stop)")
        input()
        audio_path = capture.stop()

        if audio_path:
            dataset.mark_recorded(sample.id, audio_path)
            print(f"  Saved: {audio_path.name}")
            print(f"  Progress: {dataset.recorded_count}/{dataset.total}")

    elif args.action == "train":
        mode = config.get("finetune.mode", args.mode or "offline")

        if mode == "offline":
            trainer = OfflineTrainer(
                model_name=config.get("asr.model_name", "nvidia/canary-qwen-2.5b"),
                output_dir=config.finetune_dir / "checkpoints",
                learning_rate=config.get("finetune.offline.learning_rate", 1e-4),
                num_epochs=config.get("finetune.offline.num_epochs", 5),
                batch_size=config.get("finetune.offline.batch_size", 4),
                lora_rank=config.get("finetune.offline.lora_rank", 8),
                lora_alpha=config.get("finetune.offline.lora_alpha", 16),
                max_steps=config.get("finetune.offline.max_steps", 200),
            )
            checkpoint = trainer.train(dataset)
            print(f"Fine-tuning complete! Checkpoint: {checkpoint}")
        else:
            trainer_online = OnlineTrainer(
                endpoint=config.get("finetune.online.endpoint", ""),
                api_key=config.get("finetune.online.api_key", ""),
            )
            result = trainer_online.train(dataset)
            print(f"Online fine-tuning initiated: {json.dumps(result, indent=2)}")

    elif args.action == "export":
        manifest = dataset.export_nemo_manifest()
        print(f"Manifest exported to: {manifest}")

    elif args.action == "reset":
        dataset.reset()
        print("Dataset recordings reset (sentences preserved)")


def _cmd_devices(args: argparse.Namespace) -> None:
    """List audio input devices."""
    from hola_audio.audio.capture import AudioCapture

    devices = AudioCapture.list_devices()
    if not devices:
        print("No audio input devices found.")
        return

    print("Available audio input devices:")
    for dev in devices:
        print(f"  [{dev['index']}] {dev['name']} ({dev['channels']} channels)")


def _cmd_config(args: argparse.Namespace) -> None:
    """Show or edit configuration."""
    from hola_audio.config import Config

    config = Config(args.config) if args.config else Config()

    if args.action == "show":
        import yaml

        print(yaml.safe_dump(config.as_dict(), default_flow_style=False, sort_keys=False))

    elif args.action == "get":
        value = config.get(args.key)
        if value is None:
            print(f"Key not found: {args.key}", file=sys.stderr)
            sys.exit(1)
        print(value)

    elif args.action == "set":
        config.set(args.key, args.value)
        config.save()
        print(f"Set {args.key} = {args.value}")

    elif args.action == "path":
        print(f"Config directory: {config.config_dir}")
        print(f"Data directory: {config.data_dir}")
        print(f"Recordings: {config.recordings_dir}")
        print(f"Fine-tune data: {config.finetune_dir}")

    elif args.action == "init":
        path = config.save()
        print(f"Configuration saved to: {path}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="hola-audio",
        description="Cross-platform voice input with offline speech recognition",
    )
    parser.add_argument("-c", "--config", help="Path to config file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- start -----------------------------------------------------------------
    p_start = subparsers.add_parser("start", help="Start the voice input service")
    p_start.add_argument("--no-model", action="store_true", help="Don't preload the ASR model")
    p_start.add_argument("--no-tray", action="store_true", help="Disable system tray icon")

    # -- transcribe ------------------------------------------------------------
    p_trans = subparsers.add_parser("transcribe", help="Transcribe audio file(s)")
    p_trans.add_argument("files", nargs="+", help="Audio file(s) to transcribe")
    p_trans.add_argument("--correct", action="store_true", help="Apply LLM correction")
    p_trans.add_argument("--checkpoint", help="Path to fine-tuned checkpoint")
    p_trans.add_argument("--json", action="store_true", help="Output as JSON")

    # -- correct ---------------------------------------------------------------
    p_correct = subparsers.add_parser("correct", help="Correct text via LLM")
    p_correct.add_argument("--text", help="Text to correct (or use --file/stdin)")
    p_correct.add_argument("--file", help="File containing text to correct")
    p_correct.add_argument("--context", help="Context hint for the LLM")

    # -- finetune --------------------------------------------------------------
    p_ft = subparsers.add_parser("finetune", help="Fine-tuning workflow")
    p_ft.add_argument(
        "action",
        choices=["init", "status", "record", "train", "export", "reset"],
        help="Fine-tuning action",
    )
    p_ft.add_argument("--sentences-file", help="File with custom sentences (one per line)")
    p_ft.add_argument("--mode", choices=["offline", "online"], help="Training mode")

    # -- devices ---------------------------------------------------------------
    subparsers.add_parser("devices", help="List audio input devices")

    # -- config ----------------------------------------------------------------
    p_cfg = subparsers.add_parser("config", help="Configuration management")
    p_cfg.add_argument(
        "action",
        choices=["show", "get", "set", "path", "init"],
        help="Config action",
    )
    p_cfg.add_argument("key", nargs="?", help="Config key (for get/set)")
    p_cfg.add_argument("value", nargs="?", help="Config value (for set)")

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cmd_map = {
        "start": _cmd_start,
        "transcribe": _cmd_transcribe,
        "correct": _cmd_correct,
        "finetune": _cmd_finetune,
        "devices": _cmd_devices,
        "config": _cmd_config,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
