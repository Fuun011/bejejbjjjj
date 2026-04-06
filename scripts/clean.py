import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

DEFAULT_CONFIG = {
    "bot_names": ["MEE6", "Dyno", "Rythm", "Midjourney Bot", "Groovy", "zzZTongTaiLanhLungZzz"],
    "command_prefixes": ["!", "/", "-", ".", "$"],
    "min_word_count": 2,
    "grouping_timeframe_minutes": 5,
    "max_words_per_group": 200,
    "min_words_per_group": 0,
    "log_format_regex": r'\[(.*?)\]\s(.*?):\s(.*)',
    "timestamp_format": '%Y-%m-%dT%H:%M:%S.%f%z',
    "timestamp_format_fallback": '%Y-%m-%d %H:%M:%S',
    "spam_patterns": [
        r'^[😀-🙏🌀-🗿]+$',
        r'^[<>:a-zA-Z0-9_]+$',
        r'^[👉💼💰➡️🐷🙄❤️💪]+$',
        r'^(kakaka|haha|hihi|lol|lmao|zzz)+$',
        r'^[j]{2,}$',
        r'^[=\]]+$',
        r'^[\s]+$',
    ],
    
    "low_quality_keywords": [
        "con me m", "dm", "clm", "vcl", "vl", "wtf", "cc", "đ ", " đ",
        "chiu", "haiz", "zzz"
    ],
    
    "min_content_length": 10
}

def _is_spam_or_noise(content: str, config: Dict[str, Any]) -> bool:
    """
    Check if content is spam or noise based on patterns.
    
    Args:
        content: Message content to check
        config: Configuration dictionary
        
    Returns:
        True if content is spam/noise, False otherwise
    """
    for pattern in config.get("spam_patterns", []):
        if re.match(pattern, content, re.IGNORECASE):
            return True
    
    if len(content) < config.get("min_content_length", 10):
        return True
    
    content_lower = content.lower()
    low_quality_count = sum(1 for keyword in config.get("low_quality_keywords", []) 
                           if keyword in content_lower)
    
    word_count = len(content.split())
    if word_count > 0 and (low_quality_count / word_count) > 0.3:
        return True
    
    return False

def _clean_message_content(content: str) -> str:
    """
    Cleans a single message content by removing URLs, mentions, and other noise.
    """
    content = re.sub(r'http\S+', '', content)
    content = re.sub(r'discord\.gg/\S+', '', content)
    
    content = re.sub(r'<[@#!&]\S+?>', '', content)
    
    content = re.sub(r'<a?:\w+:\d+>', '', content)
    
    content = re.sub(r'([😀-🙏🌀-🗿])\1{2,}', r'\1', content)
    
    content = re.sub(r'(.)\1{3,}', r'\1\1', content)
    
    content = re.sub(r'tenor\.com/view/\S+', '', content)
    
    content = re.sub(r'[^\w\s.,!?\'"-]', '', content)
    
    content = ' '.join(content.split())
    
    return content

def _parse_line(line: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parses a single line from the log file into a structured dictionary.
    Returns None if the line does not match the expected format.
    """
    match = re.match(config["log_format_regex"], line)
    if not match:
        return None

    try:
        timestamp_str, author, content = match.groups()
        
        try:
            timestamp = datetime.strptime(timestamp_str, config["timestamp_format"])
        except ValueError:
            timestamp = datetime.strptime(timestamp_str, config["timestamp_format_fallback"])
        
        return {"timestamp": timestamp, "author": author, "content": content}
    except (ValueError, IndexError) as e:
        return None

def process_log_file(input_file_path: str, config: Dict[str, Any] = DEFAULT_CONFIG) -> List[Dict[str, Any]]:
    """
    Processes a raw chat log file into a clean list of contextual documents.

    Args:
        input_file_path (str): The path to the input .txt log file.
        config (Dict): A configuration dictionary. Defaults to DEFAULT_CONFIG.

    Returns:
        List[Dict[str, Any]]: A list of cleaned and grouped documents, 
                                 ready for embedding. Each document is a dictionary
                                 with 'timestamp', 'author', and 'content'.
    """
    valid_messages = []
    total_lines = 0
    parsed_count = 0
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            parsed_line = _parse_line(line, config)
            if not parsed_line:
                continue
            
            parsed_count += 1

            if parsed_line["author"] in config["bot_names"]:
                continue
            if parsed_line["content"].startswith(tuple(config["command_prefixes"])):
                continue

            cleaned_content = _clean_message_content(parsed_line["content"])
            
            if _is_spam_or_noise(cleaned_content, config):
                continue
            
            if len(cleaned_content.split()) < config["min_word_count"]:
                continue
            
            words = cleaned_content.split()
            if len(words) == 1 and len(words[0]) <= 3:
                continue
            
            parsed_line["content"] = cleaned_content
            valid_messages.append(parsed_line)
    
    if len(valid_messages) > 0:
        print(f"INFO: Found {len(valid_messages)} valid messages after initial filtering.")

    if not valid_messages:
        return []

    grouped_documents = []
    current_doc = valid_messages[0].copy()
    current_word_count = len(current_doc["content"].split())

    for i in range(1, len(valid_messages)):
        next_msg = valid_messages[i]
        time_diff = next_msg["timestamp"] - current_doc["timestamp"]
        next_word_count = len(next_msg["content"].split())
        
        should_merge = (
            next_msg["author"] == current_doc["author"] and 
            time_diff <= timedelta(minutes=config["grouping_timeframe_minutes"]) and
            current_word_count + next_word_count <= config.get("max_words_per_group", 200)
        )
        
        if should_merge:
            current_doc["content"] += ". " + next_msg["content"]
            current_doc["timestamp"] = next_msg["timestamp"]
            current_word_count += next_word_count
        else:
            if current_word_count >= config.get("min_words_per_group", 10) or len(grouped_documents) == 0:
                grouped_documents.append(current_doc)
            
            current_doc = next_msg.copy()
            current_word_count = next_word_count
            
    if current_word_count >= config.get("min_words_per_group", 10) or len(grouped_documents) == 0:
        grouped_documents.append(current_doc)
    
    print(f"INFO: Grouped into {len(grouped_documents)} contextual documents.")

    for doc in grouped_documents:
        doc["timestamp"] = doc["timestamp"].isoformat()
        doc["word_count"] = len(doc["content"].split())
        doc["is_question"] = "?" in doc["content"]

    print("INFO: Processing complete.")
    return grouped_documents