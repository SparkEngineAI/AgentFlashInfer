"""
Workspace-based Knowledge Management System
Monitors workspace files and maintains per-workspace Faiss indexes for semantic search.
"""

import os
import time
import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Core dependencies
import faiss
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add tiktoken for proper token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning(
        "tiktoken not available - using word-based approximation for token counting")

logger = logging.getLogger(__name__)
# Enable debug logging for this module
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    chunk_id: str
    file_path: str
    chunk_index: int
    content: str
    token_count: int
    file_hash: str
    indexed_at: str


class WorkspaceFileMonitor(FileSystemEventHandler):
    """Monitors workspace files for changes"""

    def __init__(self, workspace_manager, workspace_id: str):
        self.workspace_manager = workspace_manager
        self.workspace_id = workspace_id
        self.debounce_delay = 2.0  # Wait 2 seconds before processing changes
        self.pending_changes = {}
        self.lock = threading.Lock()

        # Start change processor thread
        self.processor_thread = threading.Thread(
            target=self._process_changes,
            daemon=True
        )
        self.processor_thread.start()

    def on_created(self, event):
        if not event.is_directory and self._should_monitor_file(event.src_path):
            logger.info(f"File created detected: {event.src_path}")
            self._schedule_file_update(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and self._should_monitor_file(event.src_path):
            logger.info(f"File modified detected: {event.src_path}")
            self._schedule_file_update(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and self._should_monitor_file(event.src_path):
            logger.info(f"File deleted detected: {event.src_path}")
            self._schedule_file_removal(event.src_path)

    def _should_monitor_file(self, file_path: str) -> bool:
        """Check if a file should be monitored (excludes .knowledge_index directory)"""
        path_obj = Path(file_path)

        # Skip files in .knowledge_index directory to prevent infinite loops
        if '.knowledge_index' in path_obj.parts:
            logger.debug(
                f"Skipping monitoring of knowledge index file: {file_path}")
            return False

        # Skip other system/hidden directories
        for part in path_obj.parts:
            # Allow explicit check above
            if part.startswith('.') and part != '.knowledge_index':
                logger.debug(
                    f"Skipping monitoring of hidden file/directory: {file_path}")
                return False

        return True

    def _schedule_file_update(self, file_path: str):
        """Schedule a file for indexing with debouncing"""
        logger.debug(f"Scheduling file update for: {file_path}")
        with self.lock:
            self.pending_changes[file_path] = {
                'action': 'update',
                'timestamp': time.time()
            }
            logger.debug(
                f"Pending changes queue size: {len(self.pending_changes)}")

    def _schedule_file_removal(self, file_path: str):
        """Schedule a file for removal from index"""
        with self.lock:
            self.pending_changes[file_path] = {
                'action': 'remove',
                'timestamp': time.time()
            }

    def _process_changes(self):
        """Process pending file changes with debouncing"""
        while True:
            try:
                time.sleep(1)  # Check every second

                with self.lock:
                    current_time = time.time()
                    changes_to_process = []

                    for file_path, change_info in list(self.pending_changes.items()):
                        if current_time - change_info['timestamp'] >= self.debounce_delay:
                            changes_to_process.append(
                                (file_path, change_info['action']))
                            del self.pending_changes[file_path]

                # Process changes outside the lock
                if changes_to_process:
                    logger.info(
                        f"Processing {len(changes_to_process)} pending file changes")

                for file_path, action in changes_to_process:
                    try:
                        logger.info(
                            f"Processing {action} for file: {file_path}")
                        if action == 'update':
                            self.workspace_manager._update_file_index(
                                self.workspace_id, file_path
                            )
                        elif action == 'remove':
                            self.workspace_manager._remove_file_from_index(
                                self.workspace_id, file_path
                            )
                        logger.info(
                            f"Successfully processed {action} for file: {file_path}")
                    except Exception as e:
                        logger.error(
                            f"Error processing file change {action} for {file_path}: {e}")
                        import traceback
                        logger.error(
                            f"Full traceback: {traceback.format_exc()}")

            except Exception as e:
                logger.error(f"Error in file change processor: {e}")


class WorkspaceIndex:
    """Faiss-based index for a single workspace"""

    def __init__(self, workspace_path: Path, dimension: int = 1536):
        self.workspace_path = workspace_path
        self.dimension = dimension

        # Index files
        self.index_dir = workspace_path / ".knowledge_index"
        self.index_dir.mkdir(exist_ok=True)
        self.faiss_file = self.index_dir / "index.faiss"
        self.metadata_file = self.index_dir / "metadata.json"
        self.file_hashes_file = self.index_dir / "file_hashes.json"

        # Faiss index and data
        # Inner product for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[DocumentChunk] = []
        self.file_hashes: Dict[str, str] = {}
        self.lock = threading.Lock()

        self._load_index()

    def _load_index(self):
        """Load existing index from disk"""
        try:
            if (self.faiss_file.exists() and
                self.metadata_file.exists() and
                    self.file_hashes_file.exists()):

                # Load Faiss index
                self.index = faiss.read_index(str(self.faiss_file))

                # Load metadata
                with open(self.metadata_file, 'r') as f:
                    chunks_data = json.load(f)
                    self.chunks = [
                        DocumentChunk(**chunk_data) for chunk_data in chunks_data
                    ]

                # Load file hashes
                with open(self.file_hashes_file, 'r') as f:
                    self.file_hashes = json.load(f)

                logger.info(
                    f"Loaded workspace index with {len(self.chunks)} chunks")
            else:
                logger.info("No existing index found, starting fresh")

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            # Reset to clean state
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
            self.file_hashes = {}

    def _save_index(self):
        """Save index to disk"""
        try:
            logger.debug(f"Entering _save_index method")
            logger.debug(f"Attempting to acquire lock...")

            with self.lock:
                return self._save_index_unlocked()

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.debug(f"Exiting _save_index method with error")
            return False

    def _save_index_unlocked(self):
        """Save index to disk without locking (internal method)"""
        try:
            logger.debug(f"Lock acquired successfully")
            logger.debug(f"Attempting to save index to {self.index_dir}")

            # Check directory permissions
            if not os.access(self.index_dir, os.W_OK):
                logger.error(
                    f"No write permission to index directory: {self.index_dir}")
                return False

            logger.debug(f"Directory permissions OK")

            # Save Faiss index
            logger.debug(f"About to save Faiss index to {self.faiss_file}")
            logger.debug(f"Index has {self.index.ntotal} vectors")

            try:
                faiss.write_index(self.index, str(self.faiss_file))
                logger.debug(f"Faiss index write completed successfully")
            except Exception as faiss_e:
                logger.error(f"Faiss write_index failed: {faiss_e}")
                raise faiss_e

            # Save metadata
            logger.debug(
                f"Creating metadata structure with {len(self.chunks)} chunks")
            chunks_data = [
                {
                    'chunk_id': chunk.chunk_id,
                    'file_path': chunk.file_path,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'token_count': chunk.token_count,
                    'file_hash': chunk.file_hash,
                    'indexed_at': chunk.indexed_at
                }
                for chunk in self.chunks
            ]
            logger.debug(f"Metadata structure created")

            logger.debug(f"About to save metadata to {self.metadata_file}")
            try:
                with open(self.metadata_file, 'w') as f:
                    json.dump(chunks_data, f, indent=2)
                logger.debug(f"Metadata JSON write completed successfully")
            except Exception as meta_e:
                logger.error(f"Metadata write failed: {meta_e}")
                raise meta_e

            # Save file hashes
            logger.debug(
                f"About to save file hashes to {self.file_hashes_file}")
            logger.debug(f"File hashes dict: {self.file_hashes}")
            try:
                with open(self.file_hashes_file, 'w') as f:
                    json.dump(self.file_hashes, f, indent=2)
                logger.debug(f"File hashes JSON write completed successfully")
            except Exception as hash_e:
                logger.error(f"File hashes write failed: {hash_e}")
                raise hash_e

            logger.info(
                f"Successfully saved complete index with {len(self.chunks)} chunks")
            logger.debug(f"Exiting _save_index method successfully")
            return True

        except Exception as e:
            logger.error(f"Failed in _save_index_unlocked: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or word approximation"""
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.encoding_for_model(
                    "text-embedding-3-small")
                return len(encoding.encode(text))
            except Exception as e:
                logger.warning(f"Failed to use tiktoken: {e}")
                # Fall back to word approximation
                # Approximate tokens ≈ words * 1.3
                return int(len(text.split()) * 1.3)
        else:
            # Word-based approximation: tokens ≈ words * 1.3
            return int(len(text.split()) * 1.3)

    def _chunk_text(self, text: str, max_tokens: int = 7000, overlap_tokens: int = 500) -> List[str]:
        """Split text into overlapping chunks with proper token counting"""
        chunks = []
        sentences = text.replace('\n', ' ').split('. ')

        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_text = sentence + \
                ". " if not sentence.endswith('.') else sentence + " "
            sentence_tokens = self._count_tokens(sentence_text)

            # If adding this sentence would exceed the max tokens, start a new chunk
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap from end of previous chunk
                overlap_text = self._get_overlap_text(
                    current_chunk, overlap_tokens)
                current_chunk = overlap_text + sentence_text
                current_tokens = self._count_tokens(current_chunk)
            else:
                current_chunk += sentence_text
                current_tokens += sentence_tokens

        # Add the final chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Filter out chunks that are too small or too large
        filtered_chunks = []
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk)
            if 50 <= chunk_tokens <= 8000:  # Keep chunks between 50 and 8000 tokens
                filtered_chunks.append(chunk)
            elif chunk_tokens > 8000:
                # If still too large, split by paragraphs
                logger.warning(
                    f"Chunk too large ({chunk_tokens} tokens), splitting further")
                sub_chunks = self._split_large_chunk(chunk, max_tokens)
                filtered_chunks.extend(sub_chunks)

        logger.debug(
            f"Created {len(filtered_chunks)} token-limited chunks from text")
        return filtered_chunks

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get overlap text from end of chunk"""
        words = text.split()
        # Take approximately last overlap_tokens worth of words
        # Convert tokens back to approximate words
        overlap_words = int(overlap_tokens / 1.3)
        overlap_words = min(overlap_words, len(words) //
                            2)  # Don't take more than half

        if overlap_words > 0:
            return ' '.join(words[-overlap_words:]) + " "
        return ""

    def _split_large_chunk(self, text: str, max_tokens: int) -> List[str]:
        """Split a large chunk into smaller pieces"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            para_tokens = self._count_tokens(paragraph)
            current_tokens = self._count_tokens(current_chunk)

            if current_tokens + para_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _should_index_file(self, file_path: Path) -> bool:
        """Check if a file should be indexed"""

        # CRITICAL: Skip files in .knowledge_index directory to prevent infinite loops
        if '.knowledge_index' in file_path.parts:
            logger.debug(
                f"Skipping indexing of knowledge index file: {file_path}")
            return False

        # Skip other hidden files and directories (but allow explicit knowledge_index check above)
        for part in file_path.parts:
            if part.startswith('.') and part != '.knowledge_index':
                logger.debug(
                    f"Skipping indexing of hidden file/directory: {file_path}")
                return False

        # Skip binary files by extension
        text_extensions = {'.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml',
                           '.xml', '.html', '.css', '.sql', '.sh', '.bat', '.csv'}

        if file_path.suffix.lower() not in text_extensions:
            logger.debug(f"Skipping indexing of non-text file: {file_path}")
            return False

        # Skip large files (> 1MB)
        try:
            if file_path.stat().st_size > 1024 * 1024:
                logger.debug(
                    f"Skipping indexing of large file: {file_path} ({file_path.stat().st_size} bytes)")
                return False
        except:
            return False

        # Skip temporary files and system files
        if file_path.name.startswith('~') or file_path.name.endswith('.tmp') or file_path.name.endswith('.swp'):
            logger.debug(f"Skipping indexing of temporary file: {file_path}")
            return False

        return True

    def add_or_update_file(self, file_path: Path, openai_client) -> bool:
        """Add or update a file in the index"""
        try:
            logger.debug(f"add_or_update_file called for: {file_path}")

            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False

            if not self._should_index_file(file_path):
                logger.debug(
                    f"File should not be indexed (filtered out): {file_path}")
                return False

            logger.info(
                f"File passed initial checks, proceeding with indexing: {file_path}")

            # Get relative path from workspace
            try:
                rel_path = str(file_path.relative_to(self.workspace_path))
            except ValueError:
                # File not in workspace
                return False

            # Check if file changed
            current_hash = self._calculate_file_hash(file_path)
            if self.file_hashes.get(rel_path) == current_hash:
                return True  # File unchanged

            # Remove existing chunks for this file
            self._remove_file_chunks(rel_path)

            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.debug(
                    f"Successfully read file content: {len(content)} characters")
            except UnicodeDecodeError:
                logger.debug(f"Skipping binary file: {rel_path}")
                return False

            if not content.strip():
                logger.warning(
                    f"File is empty or contains only whitespace: {rel_path}")
                return False

            # Create chunks
            text_chunks = self._chunk_text(content)
            logger.debug(
                f"Created {len(text_chunks)} text chunks for file: {rel_path}")

            if not text_chunks:
                logger.warning(f"No text chunks created for file: {rel_path}")
                return False

            # Generate embeddings concurrently
            logger.info(
                f"Starting concurrent embedding generation for {len(text_chunks)} chunks")
            new_chunks, embeddings = self._generate_embeddings_concurrent(
                text_chunks, rel_path, current_hash, openai_client
            )

            if embeddings:
                logger.info(
                    f"Processing {len(embeddings)} embeddings for {rel_path}")
                with self.lock:
                    # Add to Faiss index
                    embeddings_array = np.array(embeddings)
                    logger.debug(
                        f"Adding embeddings array of shape {embeddings_array.shape} to Faiss index")
                    self.index.add(embeddings_array)
                    logger.debug(
                        f"Faiss index now has {self.index.ntotal} total vectors")

                    # Add to metadata
                    self.chunks.extend(new_chunks)
                    logger.debug(
                        f"Added {len(new_chunks)} chunks to metadata, total chunks: {len(self.chunks)}")

                    # Update file hash
                    self.file_hashes[rel_path] = current_hash
                    logger.debug(f"Updated file hash for {rel_path}")

                    # Save to disk
                    logger.debug(f"Saving index to disk...")
                    try:
                        # Try unlocked save first as a test
                        logger.debug(
                            f"Attempting unlocked save for debugging...")
                        save_success = self._save_index_unlocked()
                        if save_success:
                            logger.info(
                                f"Successfully saved index to disk (unlocked)")
                        else:
                            logger.error(
                                f"Failed to save index to disk (unlocked)")
                    except Exception as save_e:
                        logger.error(
                            f"Exception during save_index_unlocked: {save_e}")
                        import traceback
                        logger.error(
                            f"Save traceback: {traceback.format_exc()}")
                        save_success = False

                logger.info(
                    f"Successfully indexed {len(new_chunks)} chunks from {rel_path}")
                return True
            else:
                logger.warning(f"No embeddings generated for {rel_path}")
                return False

            return False

        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            return False

    def _generate_embeddings_concurrent(self, text_chunks: List[str], rel_path: str,
                                        current_hash: str, openai_client) -> Tuple[List[DocumentChunk], List[np.ndarray]]:
        """Generate embeddings for chunks concurrently"""
        new_chunks = []
        embeddings = []

        def process_chunk(chunk_info):
            """Process a single chunk (for ThreadPoolExecutor)"""
            i, chunk_text = chunk_info
            try:
                # Check token count before sending to API
                token_count = self._count_tokens(chunk_text)
                logger.debug(f"Processing chunk {i} with {token_count} tokens")

                if token_count > 8000:
                    logger.warning(
                        f"Skipping chunk {i} - too many tokens ({token_count})")
                    return None

                # Generate embedding
                logger.debug(
                    f"Generating embedding for chunk {i} of {rel_path}")
                response = openai_client.embeddings.create(
                    input=chunk_text,
                    model="text-embedding-3-small"
                )

                if not (response and response.data and response.data[0].embedding):
                    logger.warning(
                        f"Invalid embedding response for chunk {i} of {rel_path}")
                    return None

                embedding = np.array(
                    response.data[0].embedding, dtype=np.float32)
                logger.debug(
                    f"Generated embedding of dimension {embedding.shape} for chunk {i}")

                # Normalize for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)

                # Create chunk object
                chunk = DocumentChunk(
                    chunk_id=f"{rel_path}_{i}_{int(time.time())}",
                    file_path=rel_path,
                    chunk_index=i,
                    content=chunk_text,
                    token_count=token_count,
                    file_hash=current_hash,
                    indexed_at=datetime.now().isoformat()
                )

                return chunk, embedding

            except Exception as e:
                logger.error(f"Failed to process chunk {i} of {rel_path}: {e}")
                return None

        # Process chunks concurrently
        max_workers = min(10, len(text_chunks))  # Limit concurrent requests
        logger.info(
            f"Using {max_workers} workers for concurrent embedding generation")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_index = {
                executor.submit(process_chunk, (i, chunk_text)): i
                for i, chunk_text in enumerate(text_chunks)
            }

            # Collect results as they complete
            results = {}
            for future in as_completed(future_to_index):
                chunk_index = future_to_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[chunk_index] = result
                        logger.debug(
                            f"Completed embedding for chunk {chunk_index}")
                    else:
                        logger.warning(
                            f"Failed to generate embedding for chunk {chunk_index}")
                except Exception as e:
                    logger.error(f"Exception in chunk {chunk_index}: {e}")

        # Sort results by chunk index to maintain order
        for i in sorted(results.keys()):
            chunk, embedding = results[i]
            new_chunks.append(chunk)
            embeddings.append(embedding)

        logger.info(
            f"Successfully generated {len(embeddings)} embeddings concurrently for {rel_path}")
        return new_chunks, embeddings

    def _remove_file_chunks(self, rel_path: str):
        """Remove all chunks for a specific file"""
        with self.lock:
            # Find chunks to remove
            chunks_to_keep = []
            indices_to_remove = []

            for i, chunk in enumerate(self.chunks):
                if chunk.file_path == rel_path:
                    indices_to_remove.append(i)
                else:
                    chunks_to_keep.append(chunk)

            if indices_to_remove:
                # Rebuild index (Faiss doesn't support efficient removal)
                # For small workspaces this is acceptable
                self.chunks = chunks_to_keep
                self.index = faiss.IndexFlatIP(self.dimension)

                # Remove from file hashes
                self.file_hashes.pop(rel_path, None)

                logger.info(
                    f"Removed {len(indices_to_remove)} chunks for {rel_path}")

    def search(self, query_embedding: np.ndarray, k: int = 10,
               threshold: float = 0.7) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        try:
            if self.index.ntotal == 0:
                return []

            with self.lock:
                # Normalize query
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                query_norm = query_norm.reshape(1, -1).astype(np.float32)

                # Search
                scores, indices = self.index.search(
                    query_norm, min(k, self.index.ntotal))

                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if score >= threshold and 0 <= idx < len(self.chunks):
                        results.append((self.chunks[idx], float(score)))

                return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        with self.lock:
            files = set(chunk.file_path for chunk in self.chunks)
            return {
                'total_chunks': len(self.chunks),
                'total_files': len(files),
                'index_size': self.index.ntotal,
                'files': list(files)
            }


class WorkspaceKnowledgeManager:
    """Manages knowledge indexes for multiple workspaces"""

    def __init__(self, config, base_workspace_dir: str = "workspaces"):
        self.config = config
        self.base_workspace_dir = Path(base_workspace_dir)
        self.base_workspace_dir.mkdir(exist_ok=True)

        # Storage for workspace indexes and monitors
        self.workspace_indexes: Dict[str, WorkspaceIndex] = {}
        self.file_monitors: Dict[str, WorkspaceFileMonitor] = {}
        self.observers: Dict[str, Observer] = {}
        self.lock = threading.Lock()

        logger.info(
            f"WorkspaceKnowledgeManager initialized for {base_workspace_dir}")

    def get_or_create_workspace_index(self, workspace_id: str) -> WorkspaceIndex:
        """Get or create a workspace index"""
        with self.lock:
            if workspace_id not in self.workspace_indexes:
                workspace_path = self.base_workspace_dir / workspace_id
                workspace_path.mkdir(exist_ok=True)

                # Create index
                self.workspace_indexes[workspace_id] = WorkspaceIndex(
                    workspace_path)

                # Start file monitoring
                self._start_monitoring(workspace_id, workspace_path)

                # Index existing files
                self._index_existing_files(workspace_id)

                logger.info(f"Created workspace index for {workspace_id}")

            return self.workspace_indexes[workspace_id]

    def _start_monitoring(self, workspace_id: str, workspace_path: Path):
        """Start file monitoring for a workspace"""
        try:
            # Create file monitor
            file_monitor = WorkspaceFileMonitor(self, workspace_id)
            self.file_monitors[workspace_id] = file_monitor

            # Create observer
            observer = Observer()
            observer.schedule(file_monitor, str(
                workspace_path), recursive=True)
            observer.start()
            self.observers[workspace_id] = observer

            logger.info(
                f"Started file monitoring for workspace {workspace_id}")

        except Exception as e:
            logger.error(f"Failed to start monitoring for {workspace_id}: {e}")

    def _index_existing_files(self, workspace_id: str):
        """Index all existing files in a workspace"""
        try:
            if not hasattr(self.config, 'get_openai_client'):
                return

            client = self.config.get_openai_client()
            if not client:
                return

            workspace_index = self.workspace_indexes[workspace_id]
            workspace_path = self.base_workspace_dir / workspace_id

            # Find all text files
            for file_path in workspace_path.rglob('*'):
                if file_path.is_file():
                    workspace_index.add_or_update_file(file_path, client)

            logger.info(
                f"Completed indexing existing files for workspace {workspace_id}")

        except Exception as e:
            logger.error(
                f"Failed to index existing files for {workspace_id}: {e}")

    def _update_file_index(self, workspace_id: str, file_path: str):
        """Update index for a specific file"""
        try:
            logger.debug(
                f"_update_file_index called for workspace {workspace_id}, file {file_path}")

            if workspace_id not in self.workspace_indexes:
                logger.warning(
                    f"Workspace {workspace_id} not found in indexes")
                return

            client = self.config.get_openai_client()
            if not client:
                logger.error("OpenAI client not available for indexing")
                return

            workspace_index = self.workspace_indexes[workspace_id]
            file_path_obj = Path(file_path)

            logger.debug(
                f"File path object: {file_path_obj}, exists: {file_path_obj.exists()}")

            if file_path_obj.exists():
                logger.info(f"Attempting to index file: {file_path_obj}")
                success = workspace_index.add_or_update_file(
                    file_path_obj, client)
                if success:
                    logger.info(f"Successfully indexed file: {file_path_obj}")
                else:
                    logger.warning(f"Failed to index file: {file_path_obj}")
            else:
                logger.warning(f"File does not exist: {file_path_obj}")

        except Exception as e:
            logger.error(f"Failed to update file index: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

    def _remove_file_from_index(self, workspace_id: str, file_path: str):
        """Remove a file from the index"""
        try:
            if workspace_id not in self.workspace_indexes:
                return

            workspace_index = self.workspace_indexes[workspace_id]
            workspace_path = self.base_workspace_dir / workspace_id

            try:
                rel_path = str(Path(file_path).relative_to(workspace_path))
                workspace_index._remove_file_chunks(rel_path)
                workspace_index._save_index()
            except ValueError:
                pass  # File not in workspace

        except Exception as e:
            logger.error(f"Failed to remove file from index: {e}")

    def search_workspace(self, workspace_id: str, query: str,
                         max_results: int = 5, threshold: float = 0.7) -> Dict[str, Any]:
        """Search within a specific workspace"""
        try:
            if workspace_id not in self.workspace_indexes:
                return {
                    'success': False,
                    'error': f'Workspace {workspace_id} not found'
                }

            client = self.config.get_openai_client()
            if not client:
                return {
                    'success': False,
                    'error': 'OpenAI client not available'
                }

            # Generate query embedding
            response = client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )

            if not (response and response.data and response.data[0].embedding):
                return {
                    'success': False,
                    'error': 'Failed to generate query embedding'
                }

            query_embedding = np.array(response.data[0].embedding)

            # Search workspace index
            workspace_index = self.workspace_indexes[workspace_id]
            search_results = workspace_index.search(
                query_embedding, max_results, threshold)

            # Format results
            results = []
            for chunk, score in search_results:
                results.append({
                    'file_path': chunk.file_path,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'similarity_score': score,
                    'token_count': chunk.token_count
                })

            return {
                'success': True,
                'workspace_id': workspace_id,
                'query': query,
                'results': results,
                'total_matches': len(results)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get statistics for a workspace"""
        if workspace_id in self.workspace_indexes:
            return self.workspace_indexes[workspace_id].get_stats()
        return {'error': 'Workspace not found'}

    def cleanup_workspace(self, workspace_id: str):
        """Cleanup workspace resources"""
        with self.lock:
            # Stop observer
            if workspace_id in self.observers:
                self.observers[workspace_id].stop()
                self.observers[workspace_id].join()
                del self.observers[workspace_id]

            # Remove monitor
            self.file_monitors.pop(workspace_id, None)

            # Remove index
            self.workspace_indexes.pop(workspace_id, None)

            logger.info(f"Cleaned up workspace {workspace_id}")

    def shutdown(self):
        """Shutdown all workspace monitoring"""
        with self.lock:
            for observer in self.observers.values():
                observer.stop()
                observer.join()

            self.observers.clear()
            self.file_monitors.clear()
            self.workspace_indexes.clear()

            logger.info("WorkspaceKnowledgeManager shutdown complete")


# Global instance
_workspace_knowledge_manager = None


def get_workspace_knowledge_manager(config, base_workspace_dir: str = "workspaces"):
    """Get or create the global workspace knowledge manager"""
    global _workspace_knowledge_manager
    if _workspace_knowledge_manager is None:
        _workspace_knowledge_manager = WorkspaceKnowledgeManager(
            config, base_workspace_dir)
    return _workspace_knowledge_manager


def shutdown_workspace_knowledge_manager():
    """Shutdown the global workspace knowledge manager"""
    global _workspace_knowledge_manager
    if _workspace_knowledge_manager:
        _workspace_knowledge_manager.shutdown()
        _workspace_knowledge_manager = None
