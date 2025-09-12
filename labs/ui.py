"""
Beautiful terminal UI components for the Labs CLI application.
Provides modern, interactive terminal interface with rich styling, progress bars, and animations.
"""

import time
import sys
import os
from typing import Optional, Dict, Any, List, Iterator
from contextlib import contextmanager
from threading import Event, Thread
import math

from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.syntax import Syntax
from rich.rule import Rule
from rich.prompt import Prompt, Confirm
from rich.style import Style
from rich import box


class LabsUI:
    """Enhanced terminal UI for Labs CLI with beautiful styling and animations."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.current_task = None
        self._animation_stop = Event()
        
    def welcome_banner(self):
        """Display a beautiful welcome banner with ASCII art and info."""
        banner_art = """
╭─────────────────────────────────────────────────────────────╮
│  █████╗ ██╗    ██╗     █████╗ ██████╗ ███████╗              │
│ ██╔══██╗██║    ██║    ██╔══██╗██╔══██╗██╔════╝              │
│ ███████║██║    ██║    ███████║██████╔╝███████╗              │  
│ ██╔══██║██║    ██║    ██╔══██║██╔══██╗╚════██║              │
│ ██║  ██║██║    ██████╗██║  ██║██████╔╝███████║              │
│ ╚═╝  ╚═╝╚═╝    ╚═════╝╚═╝  ╚═╝╚═════╝ ╚══════╝              │
│                                                              │
│           🤖 Local LLM Inference Server                      │
│        OpenAI-Compatible • GPU-Optimized • Production-Ready │
╰─────────────────────────────────────────────────────────────╯
        """
        
        banner_panel = Panel(
            Align.center(Text(banner_art, style="bold blue")),
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        self.console.print(banner_panel)
        self.console.print()

    def show_config_summary(self, config: Dict[str, Any]):
        """Display configuration summary in a beautiful table."""
        table = Table(title="🔧 Configuration Summary", box=box.ROUNDED, show_header=True)
        table.add_column("Setting", style="cyan", no_wrap=True, width=20)
        table.add_column("Value", style="white", width=40)
        table.add_column("Status", style="green", width=10)
        
        # Model info
        model_name = config.get("model_name", "Unknown")
        table.add_row("Model", f"[bold]{model_name}[/bold]", "✓")
        
        # Generation settings
        table.add_row("Max Tokens", str(config.get("max_new_tokens", "128")), "✓")
        table.add_row("Temperature", f"{config.get('temperature', 0.7):.1f}", "✓")
        table.add_row("Top-P", f"{config.get('top_p', 0.9):.1f}", "✓")
        
        # Hardware settings
        device_map = config.get("device_map", "auto")
        dtype = str(config.get("torch_dtype", "auto")).replace("torch.", "")
        table.add_row("Device Map", device_map, "✓")
        table.add_row("Precision", dtype, "✓")
        
        
        panel = Panel(table, border_style="blue", padding=(1, 2))
        self.console.print(panel)
        self.console.print()

    @contextmanager
    def model_loading_progress(self, model_name: str):
        """Context manager for beautiful model loading progress."""
        stages = [
            "🔍 Resolving model configuration...",
            "📥 Downloading tokenizer...", 
            "⚙️  Loading model architecture...",
            "🧠 Loading model weights...",
            "🎯 Optimizing for hardware...",
            "✅ Model ready!"
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=self.console,
            expand=True
        ) as progress:
            
            # Add main task
            task_id = progress.add_task("🤖 Loading Model...", total=len(stages))
            
            def simulate_loading():
                for i, stage in enumerate(stages):
                    progress.update(task_id, description=stage, completed=i)
                    # Simulate realistic loading times
                    if "weights" in stage:
                        time.sleep(0.8)  # Model loading takes longer
                    else:
                        time.sleep(0.3)
                progress.update(task_id, completed=len(stages))
            
            # Start loading animation
            thread = Thread(target=simulate_loading)
            thread.start()
            
            try:
                yield progress
                thread.join()
            finally:
                pass
                
        # Show success message
        self.console.print(f"[green]✅ Model [bold]{model_name}[/bold] loaded successfully![/green]")
        self.console.print()

    def show_generation_header(self, prompt_type: str, streaming: bool = False):
        """Show generation header with prompt type and mode."""
        mode_icon = "🌊" if streaming else "📝"
        mode_text = "Streaming" if streaming else "Generation"
        
        header = Text()
        header.append(f"{mode_icon} {mode_text} Mode", style="bold magenta")
        header.append(" • ", style="dim")
        header.append(f"Input: {prompt_type}", style="cyan")
        
        panel = Panel(
            Align.center(header),
            border_style="magenta",
            box=box.ROUNDED
        )
        
        self.console.print(panel)
        self.console.print()

    def show_prompt_preview(self, prompt: str, max_chars: int = 200):
        """Show a preview of the input prompt."""
        if len(prompt) > max_chars:
            preview = prompt[:max_chars] + "..."
        else:
            preview = prompt
            
        prompt_panel = Panel(
            Text(preview, style="italic white"),
            title="[bold cyan]Input Prompt[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )
        
        self.console.print(prompt_panel)
        self.console.print()

    def stream_response(self, response_generator: Iterator[str]):
        """Stream response with beautiful formatting and typing effect."""
        self.console.print("[bold green]🤖 Assistant:[/bold green]")
        self.console.print()
        
        # For single cleaned text (not streaming chunks)
        if hasattr(response_generator, '__iter__'):
            response_list = list(response_generator)
            if len(response_list) == 1:
                # Single cleaned response - display directly
                response_text = response_list[0]
                response_panel = Panel(
                    Text(response_text, style="white"),
                    title="[bold green]Response[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(1, 2)
                )
                self.console.print(response_panel)
                self.console.print()
                return
        
        # Create a panel for streaming response
        response_text = Text()
        
        with Live(
            Panel(
                response_text,
                title="[bold green]Response[/bold green]",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2)
            ),
            console=self.console,
            refresh_per_second=20  # Faster refresh for better streaming
        ) as live:
            
            for chunk in response_generator:
                response_text.append(chunk, style="white")
                # Minimal delay for smooth streaming
                time.sleep(0.01)
        
        self.console.print()

    def show_response(self, response: str):
        """Display non-streaming response in a beautiful panel."""
        self.console.print("[bold green]🤖 Assistant:[/bold green]")
        self.console.print()
        
        response_panel = Panel(
            Text(response, style="white"),
            title="[bold green]Response[/bold green]", 
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print(response_panel)
        self.console.print()

    def show_stats(self, stats: Dict[str, Any]):
        """Display generation statistics in a compact table."""
        if not stats:
            return
            
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        table.add_column("Metric", style="dim cyan", no_wrap=True)
        table.add_column("Value", style="white", no_wrap=True)
        
        if "tokens_generated" in stats:
            table.add_row("Tokens Generated", str(stats["tokens_generated"]))
        if "generation_time" in stats:
            table.add_row("Generation Time", f"{stats['generation_time']:.2f}s")
        if "tokens_per_second" in stats:
            table.add_row("Tokens/Second", f"{stats['tokens_per_second']:.1f}")
        if "model_memory" in stats:
            table.add_row("GPU Memory", f"{stats['model_memory']:.1f} GB")
            
        if table.rows:
            self.console.print(Rule(style="dim"))
            self.console.print(table)

    def error_panel(self, message: str, suggestion: Optional[str] = None):
        """Display error in a beautiful, helpful panel."""
        error_text = Text()
        error_text.append("❌ Error: ", style="bold red")
        error_text.append(message, style="red")
        
        if suggestion:
            error_text.append("\n\n💡 Suggestion: ", style="bold yellow")
            error_text.append(suggestion, style="yellow")
        
        panel = Panel(
            error_text,
            title="[bold red]Error[/bold red]",
            border_style="red",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print(panel)

    def warning_panel(self, message: str):
        """Display warning message."""
        warning_text = Text()
        warning_text.append("⚠️  Warning: ", style="bold yellow")
        warning_text.append(message, style="yellow")
        
        panel = Panel(
            warning_text,
            border_style="yellow",
            box=box.ROUNDED
        )
        
        self.console.print(panel)

    def success_panel(self, message: str):
        """Display success message."""
        success_text = Text()
        success_text.append("✅ ", style="green")
        success_text.append(message, style="green")
        
        panel = Panel(
            success_text,
            border_style="green",
            box=box.ROUNDED
        )
        
        self.console.print(panel)

    def interactive_model_selection(self, available_models: List[str]) -> str:
        """Interactive model selection with beautiful interface."""
        self.console.print("[bold cyan]Available Models:[/bold cyan]")
        
        table = Table(box=box.ROUNDED, show_header=True)
        table.add_column("Index", style="cyan", width=8)
        table.add_column("Model Name", style="white")
        table.add_column("Type", style="green")
        
        for i, model in enumerate(available_models, 1):
            model_type = "Chat" if "instruct" in model.lower() or "chat" in model.lower() else "Base"
            table.add_row(str(i), model, model_type)
            
        self.console.print(table)
        self.console.print()
        
        while True:
            try:
                choice = Prompt.ask(
                    "[bold cyan]Select model (enter number)[/bold cyan]",
                    choices=[str(i) for i in range(1, len(available_models) + 1)],
                    show_choices=False
                )
                return available_models[int(choice) - 1]
            except (ValueError, IndexError):
                self.console.print("[red]Invalid selection. Please try again.[/red]")

    def confirm_action(self, message: str, default: bool = True) -> bool:
        """Beautiful confirmation prompt."""
        return Confirm.ask(f"[bold yellow]{message}[/bold yellow]", default=default)

    def input_prompt(self, message: str, default: Optional[str] = None) -> str:
        """Beautiful input prompt."""
        return Prompt.ask(f"[bold cyan]{message}[/bold cyan]", default=default)

    def show_gpu_info(self, gpu_info: Dict[str, Any]):
        """Display GPU information in a beautiful format."""
        if not gpu_info.get("available"):
            self.warning_panel("No CUDA GPU detected. Using CPU inference (slower).")
            return
            
        table = Table(title="🎮 GPU Information", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("GPU Name", gpu_info.get("name", "Unknown"))
        table.add_row("Memory Total", f"{gpu_info.get('memory_total', 0):.1f} GB")
        table.add_row("Memory Available", f"{gpu_info.get('memory_available', 0):.1f} GB")
        table.add_row("CUDA Version", gpu_info.get("cuda_version", "Unknown"))
        
        panel = Panel(table, border_style="green", padding=(1, 1))
        self.console.print(panel)
        self.console.print()

    def typing_animation(self, text: str, delay: float = 0.03):
        """Show typing animation effect."""
        for char in text:
            self.console.print(char, end="", style="green")
            time.sleep(delay)
        self.console.print()

    def progress_bar(self, items: List[str], task_name: str = "Processing"):
        """Show progress bar for batch operations."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task(f"[cyan]{task_name}...", total=len(items))
            
            for item in items:
                progress.update(task, advance=1, description=f"[cyan]{task_name}: {item}")
                yield item
                time.sleep(0.1)  # Simulate work

    def separator(self, title: Optional[str] = None):
        """Show a beautiful separator line."""
        if title:
            self.console.print(Rule(title, style="dim blue"))
        else:
            self.console.print(Rule(style="dim"))

    def print_json(self, data: Dict[str, Any], title: str = "Data"):
        """Pretty print JSON data with syntax highlighting."""
        import json
        
        json_str = json.dumps(data, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
        
        panel = Panel(
            syntax,
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan"
        )
        
        self.console.print(panel)

    def goodbye_message(self):
        """Show a beautiful goodbye message."""
        goodbye_text = Text()
        goodbye_text.append("👋 Thank you for using ", style="cyan")
        goodbye_text.append("AI Labs", style="bold blue")
        goodbye_text.append("!\n", style="cyan")
        goodbye_text.append("Happy generating! 🤖✨", style="green")
        
        panel = Panel(
            Align.center(goodbye_text),
            border_style="blue",
            box=box.DOUBLE
        )
        
        self.console.print(panel)

    @contextmanager
    def thinking_indicator(self, message: str = "🤔 Thinking..."):
        """Context manager for showing a thinking/loading indicator."""
        from rich.progress import Progress, SpinnerColumn, TextColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{message}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("thinking", total=None)
            yield progress

    def show_reasoning(self, reasoning_text: str):
        """Display the model's reasoning thoughts in a beautiful panel."""
        if not reasoning_text.strip():
            return
            
        reasoning_panel = Panel(
            Text(reasoning_text.strip(), style="dim white"),
            title="[bold yellow]🧠 Reasoning[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print(reasoning_panel)
        self.console.print()

    def stream_reasoning(self, reasoning_generator):
        """Stream reasoning thoughts in real-time with a yellow panel."""
        self.console.print("[bold yellow]🧠 Reasoning:[/bold yellow]")
        
        reasoning_text = ""
        with Live(
            Panel(
                Text("", style="dim white"),
                title="[bold yellow]Reasoning[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(1, 2)
            ),
            console=self.console,
            refresh_per_second=10
        ) as live:
            reasoning_display = Text()
            
            for chunk in reasoning_generator:
                reasoning_text += chunk
                reasoning_display.append(chunk, style="dim white")
                
                # Update the live display
                live.update(Panel(
                    reasoning_display,
                    title="[bold yellow]Reasoning[/bold yellow]",
                    border_style="yellow",
                    box=box.ROUNDED,
                    padding=(1, 2)
                ))
                
                time.sleep(0.02)  # Slightly slower for reasoning
        
        self.console.print()
        return reasoning_text


def get_terminal_width() -> int:
    """Get terminal width, with fallback."""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def is_terminal_capable() -> bool:
    """Check if terminal supports rich formatting."""
    # Always return True for now to enable rich UI
    # In production, you might want to check terminal capabilities
    return True
    # Original check:
    # return (
    #     hasattr(sys.stdout, 'isatty') and 
    #     sys.stdout.isatty() and 
    #     os.environ.get('TERM', '').lower() != 'dumb'
    # )


# Global UI instance
ui = LabsUI()

__all__ = ["LabsUI", "ui", "get_terminal_width", "is_terminal_capable"]