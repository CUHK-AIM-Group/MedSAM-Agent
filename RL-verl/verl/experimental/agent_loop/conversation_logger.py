# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Conversation Logger for Agent Loop
Records all conversations during training for debugging and analysis.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import threading

logger = logging.getLogger(__name__)


class ConversationLogger:
    """
    Logger for recording agent conversations during training.
    Thread-safe singleton implementation.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._log_dir = None
        self._log_file = None
        self._file_handle = None
        self._enabled = False
        self._conversation_count = 0
        self._write_lock = threading.Lock()

    def setup(self, log_dir: Optional[str] = None, enabled: bool = True):
        """
        Setup the conversation logger.
        
        Args:
            log_dir: Directory to save conversation logs. 
                    If None, uses './logs/conversations'
            enabled: Whether to enable logging
        """
        self._enabled = enabled
        
        if not enabled:
            logger.info("ConversationLogger is disabled")
            return

        if log_dir is None:
            log_dir = "./logs/conversations"
        
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = self._log_dir / f"conversations_{timestamp}.jsonl"
        
        # Open file handle
        self._file_handle = open(self._log_file, 'a', encoding='utf-8')
        
        logger.info(f"ConversationLogger initialized. Logging to: {self._log_file}")

    def log_conversation(
        self,
        request_id: str,
        messages: list[dict[str, Any]],
        step: Optional[int] = None,
        index: Optional[int] = None,
        extra_info: Optional[dict[str, Any]] = None,
    ):
        """
        Log a conversation.
        
        Args:
            request_id: Unique identifier for this request
            messages: Full conversation messages (including system prompt)
            step: Training step number
            index: Sample index in batch
            extra_info: Additional information to log
        """
        if not self._enabled or self._file_handle is None:
            return

        try:
            with self._write_lock:
                self._conversation_count += 1
                
                log_entry = {
                    "conversation_id": self._conversation_count,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "step": step,
                    "index": index,
                    "messages": messages,  # Complete conversation including system, user, assistant, and tool messages
                }
                
                if extra_info:
                    log_entry["extra_info"] = extra_info
                
                # Write as JSON line
                json_line = json.dumps(log_entry, ensure_ascii=False)
                self._file_handle.write(json_line + '\n')
                self._file_handle.flush()
                
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")

    def log_tool_call(
        self,
        request_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_response: str,
        success: bool,
        step: Optional[int] = None,
    ):
        """
        Log a tool call execution.
        
        Args:
            request_id: Request ID this tool call belongs to
            tool_name: Name of the tool
            tool_args: Arguments passed to the tool
            tool_response: Response from the tool
            success: Whether the tool call succeeded
            step: Training step number
        """
        if not self._enabled or self._file_handle is None:
            return

        try:
            with self._write_lock:
                log_entry = {
                    "event_type": "tool_call",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "step": step,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_response": tool_response,
                    "success": success,
                }
                
                json_line = json.dumps(log_entry, ensure_ascii=False)
                self._file_handle.write(json_line + '\n')
                self._file_handle.flush()
                
        except Exception as e:
            logger.error(f"Failed to log tool call: {e}")

    def log_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
        context: Optional[dict[str, Any]] = None,
        step: Optional[int] = None,
    ):
        """
        Log an error that occurred during conversation.
        
        Args:
            request_id: Request ID where error occurred
            error_type: Type of error
            error_message: Error message
            context: Additional context about the error
            step: Training step number
        """
        if not self._enabled or self._file_handle is None:
            return

        try:
            with self._write_lock:
                log_entry = {
                    "event_type": "error",
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "step": step,
                    "error_type": error_type,
                    "error_message": error_message,
                }
                
                if context:
                    log_entry["context"] = context
                
                json_line = json.dumps(log_entry, ensure_ascii=False)
                self._file_handle.write(json_line + '\n')
                self._file_handle.flush()
                
        except Exception as e:
            logger.error(f"Failed to log error: {e}")

    def close(self):
        """Close the log file."""
        if self._file_handle is not None:
            try:
                self._file_handle.close()
                logger.info(f"ConversationLogger closed. Total conversations logged: {self._conversation_count}")
            except Exception as e:
                logger.error(f"Failed to close log file: {e}")
            finally:
                self._file_handle = None

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


# Global singleton instance
_conversation_logger = ConversationLogger()


def get_conversation_logger() -> ConversationLogger:
    """Get the global conversation logger instance."""
    return _conversation_logger
