import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Union
from threading import Thread, Event

import torch
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

from labs import HFGenerator
from labs.config import load_config
from labs.ui import ui, is_terminal_capable
from labs.interactive import InteractiveCLI


def _parse_messages(raw: str) -> List[Dict[str, str]]:
    """
    Accept either:
      - Inline JSON: '[{"role":"user","content":"Hello"}]'
      - File reference prefixed with '@': '@/path/to/messages.json'
    """
    if raw.startswith("@"):
        path = raw[1:]
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="labs-gen",
        description="🤖 AI Labs - Beautiful terminal interface for local LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  labs-gen --prompt "Explain quantum computing" --stream
  labs-gen --messages-json '[{"role":"user","content":"Hello!"}]'
  labs-gen --prompt "Write Python code" --model codellama/CodeLlama-7b-Instruct-hf
  labs-gen --interactive  # Interactive chat mode
  labs-gen --prompt "Hello" --tts-output output.wav  # Generate speech audio
        """
    )
    # Input
    p.add_argument("--prompt", type=str, default=None, help="Raw text prompt (non-chat path).")
    p.add_argument(
        "--messages-json",
        type=str,
        default=None,
        help="Chat messages as JSON string or file reference prefixed with '@'. "
             "Example: '[{\"role\":\"user\",\"content\":\"Hello\"}]' or '@/path/to/messages.json'"
    )

    # Model/config
    p.add_argument("--model", type=str, default=None, help="Model name or path (default from config).")
    p.add_argument("--max-new-tokens", type=int, default=None, help="Maximum new tokens to generate.")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    p.add_argument("--top-p", type=float, default=None, help="Nucleus sampling p.")
    p.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    p.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty factor.")
    p.add_argument("--no-sample", action="store_true", help="Disable sampling (deterministic decoding).")

    # Streaming
    p.add_argument("--stream", action="store_true", help="Stream tokens to stdout as they are generated.")

    # Chat/template toggles
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template path even if messages are provided."
    )
    p.add_argument(
        "--no-generation-prompt",
        action="store_true",
        help="Do not add generation prompt token(s) in chat template."
    )

    # Trust remote code (for custom models)
    p.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for model/tokenizer.")


    # Config file
    p.add_argument("--config", type=str, default=None, help="Path to labs.toml for defaults (overridden by env/CLI).")
    
    # UI options
    p.add_argument("--no-ui", action="store_true", help="Disable rich terminal UI (plain output only).")
    p.add_argument("--interactive", "-i", action="store_true", help="Start interactive chat mode.")
    p.add_argument("--show-config", action="store_true", help="Display configuration and exit.")
    p.add_argument("--show-gpu", action="store_true", help="Display GPU information and exit.")

    # TTS options
    p.add_argument("--tts-output", type=str, default=None, help="Output WAV file path for text-to-speech synthesis.")

    return p


def get_gpu_info() -> Dict[str, Union[str, float, bool]]:
    """Get GPU information for display."""
    gpu_info = {"available": False}
    
    if torch.cuda.is_available():
        gpu_info["available"] = True
        gpu_info["name"] = torch.cuda.get_device_name(0)
        gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info["memory_available"] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        gpu_info["cuda_version"] = torch.version.cuda or "Unknown"
    
    return gpu_info


def estimate_tokens(text: str) -> int:
    """Rough token estimation for statistics."""
    return max(1, int(len(text.split()) * 1.3))  # Approximate


def clean_response_text(text: str) -> str:
    """Clean response text by removing unwanted tags and content."""
    import re
    
    # Remove <think> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove any remaining XML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
    text = text.strip()
    
    return text


def interactive_chat_mode(generator: HFGenerator, use_ui: bool = True) -> int:
    """Enhanced interactive chat mode with conversation management."""
    interactive_cli = InteractiveCLI(generator, use_ui)
    
    if use_ui:
        ui.console.print("[bold green]🚀 Starting Interactive Chat Mode[/bold green]")
        ui.console.print("[dim]Type /help for commands, or just start chatting![/dim]")
        ui.separator()
    else:
        print("Starting Interactive Chat Mode")
        print("Type /help for commands, or just start chatting!")
    
    try:
        while True:
            try:
                # Get user input
                if use_ui:
                    user_input = ui.input_prompt("You")
                else:
                    user_input = input("You: ")
                
                # Handle None or empty input
                if user_input is None:
                    break
                
                user_input = user_input.strip()
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    should_exit = interactive_cli.handle_command(user_input)
                    if should_exit:
                        break
                    continue
                
                # Handle regular chat
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    break
            except (EOFError, KeyboardInterrupt):
                # Handle Ctrl+C or Ctrl+D gracefully
                if use_ui:
                    ui.console.print("\n[yellow]Chat session ended by user[/yellow]")
                else:
                    print("\nChat session ended by user")
                break
            
            # Add to conversation history
            interactive_cli.history.add_message("user", user_input)
            
            # Generate response with timing
            start_time = time.time()
            conversation = interactive_cli.history.get_conversation_messages()
            
            if use_ui:
                ui.show_generation_header("Chat", streaming=True)
                
                # Stream the response with real-time display
                response_text = ""
                
                # Stream with separate reasoning and response sections
                def reasoning_response_generator():
                    """Generator that separates reasoning from final response."""
                    full_response = ""
                    reasoning_text = ""
                    response_text = ""
                    in_think_section = False
                    
                    for chunk in generator.stream_generate(
                        conversation,
                        max_new_tokens=8192,  # Very generous limit for complete responses
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    ):
                        full_response += chunk
                        
                        if "<think>" in chunk:
                            in_think_section = True
                            # Extract content before <think> as response
                            before_think = chunk.split("<think>")[0]
                            if before_think:
                                yield ("response", before_think)
                            # Start reasoning section
                            after_think = chunk.split("<think>", 1)[1]
                            if after_think:
                                yield ("reasoning", after_think)
                        elif "</think>" in chunk:
                            in_think_section = False
                            # Extract reasoning content before </think>
                            before_end = chunk.split("</think>")[0]
                            if before_end:
                                yield ("reasoning", before_end)
                            # Extract response content after </think>
                            after_end = chunk.split("</think>", 1)[1]
                            if after_end:
                                yield ("response", after_end)
                        elif in_think_section:
                            # We're inside thinking tags
                            yield ("reasoning", chunk)
                        else:
                            # Regular response content
                            yield ("response", chunk)
                    
                    return full_response
                
                # Process the stream with separate displays
                reasoning_chunks = []
                response_chunks = []
                reasoning_started = False
                response_started = False
                
                for section_type, chunk in reasoning_response_generator():
                    if section_type == "reasoning":
                        reasoning_chunks.append(chunk)
                        if not reasoning_started:
                            # Start reasoning display
                            ui.console.print("[bold yellow]🧠 Reasoning:[/bold yellow]")
                            reasoning_started = True
                        ui.console.print(chunk, end="", style="dim yellow")
                    
                    elif section_type == "response":
                        response_chunks.append(chunk)
                        if not response_started and reasoning_started:
                            # Finish reasoning section and start response
                            ui.console.print("\n")
                            ui.separator("Final Answer")
                            ui.console.print("[bold green]🤖 Assistant:[/bold green]")
                            ui.console.print()
                            response_started = True
                        elif not response_started:
                            # Direct response without reasoning
                            ui.console.print("[bold green]🤖 Assistant:[/bold green]")
                            ui.console.print()
                            response_started = True
                        
                        ui.console.print(chunk, end="", style="white")
                
                ui.console.print()  # New line after streaming
                ui.console.print()  # Extra space
                
                # Store the actual response (without reasoning) for history
                response_text = "".join(response_chunks)
            else:
                # Non-UI mode with reasoning separation
                reasoning_chunks = []
                response_chunks = []
                reasoning_started = False
                response_started = False
                
                def reasoning_response_generator_simple():
                    """Simple generator for non-UI mode."""
                    in_think_section = False
                    
                    for chunk in generator.stream_generate(
                        conversation,
                        max_new_tokens=8192,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    ):
                        if "<think>" in chunk:
                            in_think_section = True
                            before_think = chunk.split("<think>")[0]
                            if before_think:
                                yield ("response", before_think)
                            after_think = chunk.split("<think>", 1)[1]
                            if after_think:
                                yield ("reasoning", after_think)
                        elif "</think>" in chunk:
                            in_think_section = False
                            before_end = chunk.split("</think>")[0]
                            if before_end:
                                yield ("reasoning", before_end)
                            after_end = chunk.split("</think>", 1)[1]
                            if after_end:
                                yield ("response", after_end)
                        elif in_think_section:
                            yield ("reasoning", chunk)
                        else:
                            yield ("response", chunk)
                
                for section_type, chunk in reasoning_response_generator_simple():
                    if section_type == "reasoning":
                        reasoning_chunks.append(chunk)
                        if not reasoning_started:
                            print("🧠 Reasoning:")
                            reasoning_started = True
                        print(chunk, end="", flush=True)
                    
                    elif section_type == "response":
                        response_chunks.append(chunk)
                        if not response_started and reasoning_started:
                            print("\n\n--- Final Answer ---")
                            print("Assistant: ", end="", flush=True)
                            response_started = True
                        elif not response_started:
                            print("Assistant: ", end="", flush=True)
                            response_started = True
                        
                        print(chunk, end="", flush=True)
                
                print()  # New line after streaming
                response_text = "".join(response_chunks)
            
            # Update conversation and stats
            generation_time = time.time() - start_time
            tokens_generated = estimate_tokens(response_text)
            
            interactive_cli.history.add_message("assistant", response_text)
            interactive_cli.session_stats["messages_sent"] += 1
            interactive_cli.session_stats["tokens_generated"] += tokens_generated
            interactive_cli.session_stats["total_time"] += generation_time
            
            # Show stats
            if use_ui:
                stats = {
                    "tokens_generated": tokens_generated,
                    "generation_time": generation_time,
                    "tokens_per_second": tokens_generated / generation_time if generation_time > 0 else 0
                }
                ui.show_stats(stats)
                ui.separator()
    
    except KeyboardInterrupt:
        if use_ui:
            ui.console.print("\n[yellow]Chat interrupted by user[/yellow]")
        else:
            print("\nChat interrupted by user")
    
    # Ask if user wants to save conversation
    if interactive_cli.history.current_session:
        save_conversation = False
        if use_ui:
            save_conversation = ui.confirm_action("Save this conversation?", default=True)
        else:
            response = input("Save this conversation? (Y/n): ")
            save_conversation = not response.lower().startswith('n')
        
        if save_conversation:
            interactive_cli.save_conversation()
    
    if use_ui:
        ui.goodbye_message()
    else:
        print("Goodbye!")
    
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    
    # Determine if we should use rich UI
    use_ui = not args.no_ui and is_terminal_capable()
    
    # Show welcome banner
    if use_ui:
        ui.welcome_banner()
    
    # Load configuration
    try:
        cfg = load_config(args.config)
    except Exception as e:
        if use_ui:
            ui.error_panel(f"Failed to load configuration: {e}")
        else:
            print(f"Error: Failed to load configuration: {e}", file=sys.stderr)
        return 2
    
    # Handle special modes
    if args.show_gpu:
        gpu_info = get_gpu_info()
        if use_ui:
            ui.show_gpu_info(gpu_info)
        else:
            print(f"GPU Available: {gpu_info['available']}")
            if gpu_info['available']:
                print(f"GPU Name: {gpu_info['name']}")
                print(f"Total Memory: {gpu_info['memory_total']:.1f} GB")
        return 0
    
    if args.show_config:
        config_dict = {
            "model_name": cfg.model_name,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "device_map": cfg.device_map,
            "torch_dtype": str(cfg.torch_dtype),
        }
        if use_ui:
            ui.show_config_summary(config_dict)
        else:
            for key, value in config_dict.items():
                print(f"{key}: {value}")
        return 0
    
    # Validate input arguments
    if not args.interactive and not args.prompt and not args.messages_json and not args.tts_output:
        if use_ui:
            ui.error_panel(
                "No input provided", 
                "Use --prompt for text, --messages-json for chat, --tts-output for TTS, or --interactive for chat mode"
            )
        else:
            print("Error: Provide either --prompt, --messages-json, --tts-output, or --interactive", file=sys.stderr)
        return 2

    # Apply CLI overrides to config
    if args.model:
        cfg.model_name = args.model
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.repetition_penalty is not None:
        cfg.repetition_penalty = args.repetition_penalty
    if args.no_chat_template:
        cfg.use_chat_template = False
    if args.no_generation_prompt:
        cfg.add_generation_prompt = False
    if args.trust_remote_code:
        cfg.trust_remote_code = True

    # Show configuration summary
    if use_ui:
        config_dict = {
            "model_name": cfg.model_name,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "device_map": cfg.device_map,
            "torch_dtype": str(cfg.torch_dtype),
        }
        ui.show_config_summary(config_dict)

    # Initialize generator with beautiful loading animation
    if use_ui:
        with ui.model_loading_progress(cfg.model_name):
            try:
                gen = HFGenerator(cfg)
            except Exception as e:
                ui.error_panel(
                    f"Failed to load model: {e}",
                    "Try enabling quantization (--load-in-4bit) or using a smaller model"
                )
                return 2
    else:
        try:
            print(f"Loading model: {cfg.model_name}...", file=sys.stderr)
            gen = HFGenerator(cfg)
            print("Model loaded successfully!", file=sys.stderr)
        except Exception as e:
            print(f"Error: Failed to load model: {e}", file=sys.stderr)
            return 2

    # Handle TTS mode
    if args.tts_output:
        # Use prompt or messages_json as input text
        if args.prompt:
            text_input = args.prompt
        elif args.messages_json:
            try:
                messages = _parse_messages(args.messages_json)
                text_input = " ".join(msg["content"] for msg in messages if "content" in msg)
            except Exception as e:
                error_msg = f"Error parsing messages JSON: {e}"
                if use_ui:
                    ui.error_panel(error_msg, "Check JSON format: [{\"role\":\"user\",\"content\":\"...\"}]")
                else:
                    print(error_msg, file=sys.stderr)
                return 2
        else:
            if use_ui:
                ui.error_panel("No input text provided for TTS")
            else:
                print("Error: No input text provided for TTS", file=sys.stderr)
            return 2

        # Lazy import TTS only when needed
        try:
            from labs.tts import get_tts_instance
        except Exception as e:
            if use_ui:
                ui.error_panel(
                    f"Failed to initialize TTS module: {e}",
                    "Ensure TTS dependencies are installed or switch to non-TTS mode"
                )
            else:
                print(f"Error: Failed to initialize TTS module: {e}", file=sys.stderr)
            return 2

        tts = get_tts_instance()
        try:
            audio_bytes = tts.synthesize(text_input)
            with open(args.tts_output, "wb") as f:
                f.write(audio_bytes)
            if use_ui:
                ui.console.print(f"[green]Audio saved to {args.tts_output}[/green]")
            else:
                print(f"Audio saved to {args.tts_output}")
            return 0
        except Exception as e:
            if use_ui:
                ui.error_panel(f"TTS synthesis failed: {e}")
            else:
                print(f"Error: TTS synthesis failed: {e}", file=sys.stderr)
            return 2

    # Handle interactive chat mode
    if args.interactive:
        return interactive_chat_mode(gen, use_ui)

    # Build input
    prompt_or_messages: Union[str, List[Dict[str, str]]]
    prompt_type = "Unknown"
    
    if args.messages_json and not args.no_chat_template:
        try:
            prompt_or_messages = _parse_messages(args.messages_json)
            prompt_type = "Chat Messages"
        except Exception as e:
            error_msg = f"Error parsing messages JSON: {e}"
            if use_ui:
                ui.error_panel(error_msg, "Check JSON format: [{\"role\":\"user\",\"content\":\"...\"}]")
            else:
                print(error_msg, file=sys.stderr)
            return 2
    elif args.prompt:
        prompt_or_messages = args.prompt
        prompt_type = "Text Prompt"
    else:
        # If messages provided but user disabled chat template, coerce to string
        try:
            parsed = _parse_messages(args.messages_json)  # type: ignore[arg-type]
            prompt_or_messages = json.dumps(parsed, ensure_ascii=False)
            prompt_type = "Raw JSON"
        except Exception:
            prompt_or_messages = str(args.messages_json)
            prompt_type = "Raw Text"
    
    # Show generation header and prompt preview
    if use_ui:
        ui.show_generation_header(prompt_type, args.stream)
        if isinstance(prompt_or_messages, str):
            ui.show_prompt_preview(prompt_or_messages)
        else:
            # Show chat messages preview
            chat_preview = "\n".join([f"{msg['role']}: {msg['content'][:100]}" for msg in prompt_or_messages[-2:]])
            ui.show_prompt_preview(chat_preview)

    # Generate with timing and statistics
    start_time = time.time()
    
    try:
        if args.stream:
            if use_ui:
                # Collect chunks for UI streaming
                chunks = []
                for chunk in gen.stream_generate(
                    prompt_or_messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=(not args.no_sample),
                    repetition_penalty=args.repetition_penalty,
                ):
                    chunks.append(chunk)
                
                generated_text = "".join(chunks)
                # Clean the response text
                generated_text = clean_response_text(generated_text)
                
                ui.stream_response(iter([generated_text]))
            else:
                # Plain streaming output
                generated_text = ""
                for chunk in gen.stream_generate(
                    prompt_or_messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=(not args.no_sample),
                    repetition_penalty=args.repetition_penalty,
                ):
                    generated_text += chunk
                
                # Clean the response text
                generated_text = clean_response_text(generated_text)
                print(generated_text)
        else:
            # Non-streaming generation
            generated_text = gen.generate(
                prompt_or_messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=(not args.no_sample),
                repetition_penalty=args.repetition_penalty,
            )
            
            # Clean the response text
            generated_text = clean_response_text(generated_text)
            
            if use_ui:
                ui.show_response(generated_text)
            else:
                print(generated_text)
        
        # Show generation statistics
        if use_ui:
            generation_time = time.time() - start_time
            tokens_generated = estimate_tokens(generated_text)
            
            stats = {
                "tokens_generated": tokens_generated,
                "generation_time": generation_time,
                "tokens_per_second": tokens_generated / generation_time if generation_time > 0 else 0
            }
            
            # Add GPU memory info if available
            if torch.cuda.is_available():
                stats["model_memory"] = torch.cuda.memory_allocated() / 1024**3
            
            ui.show_stats(stats)
        
        return 0
        
    except KeyboardInterrupt:
        if use_ui:
            ui.warning_panel("Generation interrupted by user")
        else:
            print("\nGeneration interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        if use_ui:
            ui.error_panel(
                f"Generation failed: {e}",
                "Try reducing max_new_tokens or enabling quantization"
            )
        else:
            print(f"Error: Generation failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        if is_terminal_capable():
            ui.warning_panel("Application interrupted by user")
        else:
            print("\nApplication interrupted by user", file=sys.stderr)
        exit_code = 1
    except Exception as e:
        if is_terminal_capable():
            ui.error_panel(f"Unexpected error: {e}")
        else:
            print(f"Error: {e}", file=sys.stderr)
        exit_code = 2
    
    raise SystemExit(exit_code)
