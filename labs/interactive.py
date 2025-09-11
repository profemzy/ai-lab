"""
Interactive features and utilities for the Labs CLI application.
Provides conversation management, chat history, and advanced interactive modes.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box

from labs.ui import ui


class ConversationHistory:
    """Manages conversation history with save/load functionality."""
    
    def __init__(self, history_dir: Optional[str] = None):
        self.history_dir = Path(history_dir or os.path.expanduser("~/.labs/conversations"))
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.conversations = []
        self.current_session = []
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the current session."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.current_session.append(message)
    
    def save_conversation(self, title: Optional[str] = None) -> str:
        """Save the current conversation to disk."""
        if not self.current_session:
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        title = title or f"conversation_{timestamp}"
        filename = f"{title}.json"
        filepath = self.history_dir / filename
        
        conversation_data = {
            "title": title,
            "created_at": datetime.now().isoformat(),
            "messages": self.current_session.copy(),
            "stats": {
                "message_count": len(self.current_session),
                "user_messages": len([m for m in self.current_session if m["role"] == "user"]),
                "assistant_messages": len([m for m in self.current_session if m["role"] == "assistant"])
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def load_conversation(self, filename: str) -> List[Dict[str, Any]]:
        """Load a conversation from disk."""
        filepath = self.history_dir / filename
        if not filepath.exists():
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("messages", [])
        except (json.JSONDecodeError, KeyError):
            return []
    
    def list_conversations(self) -> List[Dict[str, str]]:
        """List all saved conversations."""
        conversations = []
        
        for filepath in self.history_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                conversations.append({
                    "filename": filepath.name,
                    "title": data.get("title", filepath.stem),
                    "created_at": data.get("created_at", "Unknown"),
                    "message_count": data.get("stats", {}).get("message_count", 0)
                })
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Sort by creation time, newest first
        conversations.sort(key=lambda x: x["created_at"], reverse=True)
        return conversations
    
    def clear_session(self):
        """Clear the current session."""
        self.current_session = []
    
    def get_conversation_messages(self) -> List[Dict[str, str]]:
        """Get messages in format suitable for model generation."""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.current_session]


class InteractiveCLI:
    """Enhanced interactive CLI with conversation management and advanced features."""
    
    def __init__(self, generator, use_ui: bool = True):
        self.generator = generator
        self.use_ui = use_ui
        self.history = ConversationHistory()
        self.session_stats = {
            "messages_sent": 0,
            "tokens_generated": 0,
            "total_time": 0.0
        }
    
    def show_help(self):
        """Display help information."""
        if self.use_ui:
            help_text = """
[bold cyan]Interactive Mode Commands:[/bold cyan]

[yellow]/help[/yellow]           - Show this help message
[yellow]/save[/yellow]           - Save current conversation  
[yellow]/load[/yellow]           - Load a saved conversation
[yellow]/list[/yellow]           - List saved conversations
[yellow]/clear[/yellow]          - Clear current conversation
[yellow]/stats[/yellow]          - Show session statistics
[yellow]/config[/yellow]         - Show current model configuration
[yellow]/model[/yellow]          - Switch model (if supported)
[yellow]/exit, /quit[/yellow]    - Exit interactive mode

[dim]Or just type your message to chat with the AI![/dim]
            """
            ui.console.print(help_text)
        else:
            print("Commands: /help, /save, /load, /list, /clear, /stats, /config, /exit")
    
    def save_conversation(self):
        """Save the current conversation."""
        if not self.history.current_session:
            if self.use_ui:
                ui.warning_panel("No conversation to save")
            else:
                print("No conversation to save")
            return
        
        title = None
        if self.use_ui:
            title = ui.input_prompt("Enter conversation title (optional)")
        
        filepath = self.history.save_conversation(title if title else None)
        
        if self.use_ui:
            ui.success_panel(f"Conversation saved: {filepath}")
        else:
            print(f"Conversation saved: {filepath}")
    
    def load_conversation(self):
        """Load a saved conversation."""
        conversations = self.history.list_conversations()
        
        if not conversations:
            if self.use_ui:
                ui.warning_panel("No saved conversations found")
            else:
                print("No saved conversations found")
            return
        
        if self.use_ui:
            # Show conversations in a table
            table = Table(title="Saved Conversations", box=box.ROUNDED)
            table.add_column("Index", style="cyan", width=8)
            table.add_column("Title", style="white", width=30)
            table.add_column("Date", style="green", width=20)
            table.add_column("Messages", style="yellow", width=10)
            
            for i, conv in enumerate(conversations, 1):
                created_date = conv["created_at"][:10] if len(conv["created_at"]) > 10 else conv["created_at"]
                table.add_row(str(i), conv["title"], created_date, str(conv["message_count"]))
            
            ui.console.print(table)
            
            try:
                choice = int(ui.input_prompt("Enter conversation number")) - 1
                if 0 <= choice < len(conversations):
                    filename = conversations[choice]["filename"]
                    messages = self.history.load_conversation(filename)
                    if messages:
                        self.history.current_session = messages
                        ui.success_panel(f"Loaded conversation: {conversations[choice]['title']}")
                    else:
                        ui.error_panel("Failed to load conversation")
                else:
                    ui.error_panel("Invalid selection")
            except (ValueError, IndexError):
                ui.error_panel("Invalid input")
        else:
            # Simple list for non-UI mode
            for i, conv in enumerate(conversations, 1):
                print(f"{i}. {conv['title']} ({conv['message_count']} messages)")
            
            try:
                choice = int(input("Enter conversation number: ")) - 1
                if 0 <= choice < len(conversations):
                    filename = conversations[choice]["filename"]
                    messages = self.history.load_conversation(filename)
                    if messages:
                        self.history.current_session = messages
                        print(f"Loaded conversation: {conversations[choice]['title']}")
                    else:
                        print("Failed to load conversation")
            except (ValueError, IndexError):
                print("Invalid selection")
    
    def show_stats(self):
        """Display session statistics."""
        stats = self.session_stats.copy()
        stats["conversation_length"] = len(self.history.current_session)
        stats["avg_response_time"] = (
            stats["total_time"] / max(1, stats["messages_sent"])
        )
        
        if self.use_ui:
            ui.show_stats(stats)
        else:
            print(f"Messages sent: {stats['messages_sent']}")
            print(f"Tokens generated: {stats['tokens_generated']}")
            print(f"Total time: {stats['total_time']:.2f}s")
            print(f"Average response time: {stats['avg_response_time']:.2f}s")
    
    def clear_conversation(self):
        """Clear the current conversation."""
        if not self.history.current_session:
            if self.use_ui:
                ui.warning_panel("No conversation to clear")
            else:
                print("No conversation to clear")
            return
        
        confirm = True
        if self.use_ui:
            confirm = ui.confirm_action("Clear current conversation?", default=False)
        else:
            response = input("Clear current conversation? (y/N): ")
            confirm = response.lower().startswith('y')
        
        if confirm:
            self.history.clear_session()
            self.session_stats = {"messages_sent": 0, "tokens_generated": 0, "total_time": 0.0}
            
            if self.use_ui:
                ui.success_panel("Conversation cleared")
            else:
                print("Conversation cleared")
    
    def show_config(self):
        """Show current model configuration."""
        config_dict = {
            "model_name": self.generator.config.model_name,
            "max_new_tokens": self.generator.config.max_new_tokens,
            "temperature": self.generator.config.temperature,
            "top_p": self.generator.config.top_p,
            "device_map": self.generator.config.device_map,
            "torch_dtype": str(self.generator.config.torch_dtype),
        }
        
        if self.use_ui:
            ui.show_config_summary(config_dict)
        else:
            for key, value in config_dict.items():
                print(f"{key}: {value}")
    
    def list_conversations(self):
        """List all saved conversations."""
        conversations = self.history.list_conversations()
        
        if not conversations:
            if self.use_ui:
                ui.warning_panel("No saved conversations found")
            else:
                print("No saved conversations found")
            return
        
        if self.use_ui:
            table = Table(title="Saved Conversations", box=box.ROUNDED)
            table.add_column("Title", style="white", width=30)
            table.add_column("Date", style="green", width=20)
            table.add_column("Messages", style="yellow", width=10)
            
            for conv in conversations:
                created_date = conv["created_at"][:10] if len(conv["created_at"]) > 10 else conv["created_at"]
                table.add_row(conv["title"], created_date, str(conv["message_count"]))
            
            ui.console.print(table)
        else:
            for conv in conversations:
                print(f"{conv['title']} - {conv['created_at'][:10]} ({conv['message_count']} messages)")
    
    def handle_command(self, command: str) -> bool:
        """Handle interactive commands. Returns True if command was handled, False to continue."""
        command = command.strip().lower()
        
        if command in ['/help', '/h']:
            self.show_help()
        elif command in ['/save', '/s']:
            self.save_conversation()
        elif command in ['/load', '/l']:
            self.load_conversation()
        elif command in ['/list', '/ls']:
            self.list_conversations()
        elif command in ['/clear', '/c']:
            self.clear_conversation()
        elif command in ['/stats', '/st']:
            self.show_stats()
        elif command in ['/config', '/cfg']:
            self.show_config()
        elif command in ['/exit', '/quit', '/q']:
            return True  # Signal to exit
        else:
            if self.use_ui:
                ui.error_panel(f"Unknown command: {command}", "Type /help for available commands")
            else:
                print(f"Unknown command: {command}. Type /help for available commands.")
        
        return False


__all__ = ["ConversationHistory", "InteractiveCLI"]